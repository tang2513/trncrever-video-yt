"""Processamento de áudio com FFmpeg e divisão em blocos com continuidade exata.

Responsabilidades:

* Localizar e verificar o FFmpeg/FFprobe (PATH ou locais comuns no Windows).
* Inspecionar a mídia (duração, faixas de áudio) e detectar arquivos corrompidos.
* Extrair o áudio direto do disco para WAV PCM 16 bits, mono, 16 kHz — o
  arquivo original é apenas LIDO; o FFmpeg processa em fluxo (pouca RAM, mesmo
  com vídeos de ~10 GB). Progresso real a partir de ``-progress pipe:1``.
* Detectar pausas com ``silencedetect`` SOMENTE em janelas próximas a cada
  limite nominal de bloco (rápido mesmo para 4 horas de áudio).
* Planejar os cortes: ~10 min por bloco, ajustados para a pausa mais adequada
  dentro de uma janela configurável; sem pausa adequada → corte no limite nominal.
* Gravar os blocos no disco copiando amostras do WAV com o módulo ``wave``.

Por que cortar por número de amostras (e não com ``-ss``/``-t`` em segundos)?
Os limites de cada bloco são inteiros de amostra: ``fim(bloco N) == início(bloco
N+1)``. Isso garante, por construção, que nenhum trecho é descartado, duplicado
ou deslocado — sem precisar de deduplicação de texto. O offset de cada bloco
(``start_sample / 16000``) converte os timestamps do bloco para o tempo original.

A detecção de silêncio é usada APENAS para escolher onde cortar; nenhum trecho
de áudio (nem fala em volume baixo) é removido.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .utils import (
    IS_WINDOWS,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLE_WIDTH,
    AudioProcessingError,
    CorruptedMediaError,
    DiskSpaceError,
    FFmpegNotFoundError,
    TaskCancelledError,
    TranscriberError,
    ensure_disk_space,
    estimate_wav_bytes,
    run_process,
)

logger = logging.getLogger(__name__)

CancelCheck = Callable[[], bool]

DEFAULT_CHUNK_SECONDS = 600.0      # ~10 minutos por bloco
DEFAULT_SEARCH_WINDOW = 60.0       # procura pausas até ±60 s do limite nominal
DEFAULT_SILENCE_NOISE_DB = -35.0   # silencedetect=noise=-35dB
DEFAULT_SILENCE_MIN_DURATION = 0.4  # silencedetect=d=0.4

_COMMON_FFMPEG_DIRS_WIN = (
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
    r"C:\tools\ffmpeg\bin",
)


# =============================================================================
# FFmpeg
# =============================================================================


@dataclass(frozen=True)
class FFmpegInfo:
    ffmpeg: str
    ffprobe: str
    version: str

    @property
    def bin_dir(self) -> str:
        return str(Path(self.ffmpeg).parent)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _which(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    env_dir = os.environ.get("FFMPEG_DIR") or os.environ.get("FFMPEG_BIN")
    dirs = ([env_dir] if env_dir else []) + (list(_COMMON_FFMPEG_DIRS_WIN) if IS_WINDOWS else [])
    exe = f"{name}.exe" if IS_WINDOWS else name
    for d in dirs:
        candidate = Path(d) / exe
        if candidate.is_file():
            return str(candidate)
    return None


def find_ffmpeg() -> FFmpegInfo:
    """Localiza FFmpeg e FFprobe e confirma que executam. Lança ``FFmpegNotFoundError``."""
    ffmpeg = _which("ffmpeg")
    ffprobe = _which("ffprobe")
    if not ffmpeg:
        raise FFmpegNotFoundError()
    if not ffprobe:
        raise FFmpegNotFoundError(
            "O FFprobe (que acompanha o FFmpeg) não foi encontrado. Reinstale o FFmpeg completo e "
            "verifique o PATH."
        )
    try:
        result = run_process([ffmpeg, "-hide_banner", "-version"], timeout=30)
    except TranscriberError as exc:
        raise FFmpegNotFoundError(technical=exc.technical) from exc
    if result.returncode != 0 or not result.stdout_tail:
        raise FFmpegNotFoundError(
            "O FFmpeg foi encontrado, mas não pôde ser executado.", technical=result.stderr_text[-800:]
        )
    first = result.stdout_tail[0]
    match = re.search(r"ffmpeg version (\S+)", first)
    return FFmpegInfo(ffmpeg=ffmpeg, ffprobe=ffprobe, version=match.group(1) if match else first.strip())


def check_ffmpeg() -> tuple[bool, str]:
    """Versão "sem exceção" para a interface: ``(ok, mensagem)``."""
    try:
        info = find_ffmpeg()
    except FFmpegNotFoundError as exc:
        return False, exc.user_message
    return True, f"FFmpeg {info.version} encontrado em {info.bin_dir}"


# =============================================================================
# Inspeção de mídia
# =============================================================================


@dataclass
class MediaInfo:
    path: str
    duration: Optional[float]
    size_bytes: int
    format_name: str
    has_video: bool
    audio_streams: int
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(value: Any) -> Optional[float]:
    try:
        v = float(value)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def probe_media(path: Path | str, ffmpeg: Optional[FFmpegInfo] = None, *, timeout: float = 120) -> MediaInfo:
    """Lê as informações do arquivo com ``ffprobe``. Detecta corrupção e ausência de áudio."""
    ffmpeg = ffmpeg or find_ffmpeg()
    path = Path(path)
    out: list[str] = []
    result = run_process(
        [ffmpeg.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        on_stdout_line=out.append,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise CorruptedMediaError(technical=result.stderr_text[-1500:])
    try:
        data = json.loads("\n".join(out) or "{}")
    except json.JSONDecodeError as exc:
        raise CorruptedMediaError(technical=f"ffprobe JSON inválido: {exc}") from exc

    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    video = [s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")]
    if not audio:
        raise CorruptedMediaError("Este arquivo não contém nenhuma faixa de áudio para transcrever.")

    duration = _f(fmt.get("duration")) or _f(audio[0].get("duration"))
    a0 = audio[0]
    return MediaInfo(
        path=str(path),
        duration=duration,
        size_bytes=path.stat().st_size if path.exists() else int(fmt.get("size") or 0),
        format_name=str(fmt.get("format_name") or ""),
        has_video=bool(video),
        audio_streams=len(audio),
        audio_codec=a0.get("codec_name"),
        audio_sample_rate=int(a0["sample_rate"]) if str(a0.get("sample_rate", "")).isdigit() else None,
        audio_channels=a0.get("channels"),
    )


# =============================================================================
# WAV de trabalho
# =============================================================================


@dataclass
class WavInfo:
    path: str
    sample_rate: int
    channels: int
    sample_width: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / float(self.sample_rate) if self.sample_rate else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration"] = self.duration
        return d


def read_wav_info(path: Path | str) -> WavInfo:
    """Lê o cabeçalho do WAV e confirma o formato de trabalho (16 kHz, mono, 16 bits)."""
    try:
        with wave.open(str(path), "rb") as w:
            info = WavInfo(str(path), w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioProcessingError("O áudio convertido está incompleto ou corrompido.", technical=repr(exc)) from exc
    if (info.sample_rate, info.channels, info.sample_width) != (TARGET_SAMPLE_RATE, TARGET_CHANNELS, TARGET_SAMPLE_WIDTH):
        raise AudioProcessingError(
            "O áudio convertido não está no formato esperado.",
            technical=f"rate={info.sample_rate} ch={info.channels} width={info.sample_width}",
        )
    return info


def _is_valid_wav(path: Path) -> bool:
    try:
        return path.is_file() and read_wav_info(path).frames > 0
    except AudioProcessingError:
        return False


_FFMPEG_ERROR_HINTS: list[tuple[tuple[str, ...], type[TranscriberError], str]] = [
    (("no space left", "disk full", "not enough space"), DiskSpaceError, DiskSpaceError.default_message),
    (("invalid data found", "moov atom not found", "could not find codec parameters", "invalid argument",
      "error while decoding", "corrupt"), CorruptedMediaError, CorruptedMediaError.default_message),
    (("matches no streams", "does not contain any stream", "output file is empty"), CorruptedMediaError,
     "Este arquivo não contém uma faixa de áudio que possa ser convertida."),
    (("permission denied",), AudioProcessingError,
     "Sem permissão para ler o arquivo ou gravar na pasta temporária."),
]


def _ffmpeg_error(stderr: str, default: str) -> TranscriberError:
    low = (stderr or "").lower()
    for keys, cls, msg in _FFMPEG_ERROR_HINTS:
        if any(k in low for k in keys):
            return cls(msg, technical=stderr[-1500:])
    return AudioProcessingError(default, technical=(stderr or "")[-1500:])


def extract_audio(
    source: Path | str,
    dest_wav: Path | str,
    *,
    ffmpeg: Optional[FFmpegInfo] = None,
    duration: Optional[float] = None,
    progress_cb: Optional[Callable[[Optional[float], float], None]] = None,
    cancel_check: Optional[CancelCheck] = None,
    on_process_start: Optional[Callable[[int], None]] = None,
    on_process_exit: Optional[Callable[[int], None]] = None,
    reserve_for_chunks: bool = True,
) -> WavInfo:
    """Converte a primeira faixa de áudio de ``source`` em WAV 16 kHz mono PCM 16 bits.

    * O resultado é gravado em ``*.partial.wav`` e só depois renomeado: um WAV
      final existente é sempre completo (permite retomar sem reconverter).
    * ``progress_cb(fração | None, segundos_processados)`` — fração real com base
      na duração; ``None`` se a duração for desconhecida.
    * Verifica espaço em disco antes (WAV + cópia em blocos, se ``reserve_for_chunks``).
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    source = Path(source)
    dest = Path(dest_wav)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if source.resolve() == dest.resolve():
        raise AudioProcessingError("O arquivo de destino não pode ser o próprio arquivo original.")

    if _is_valid_wav(dest):
        logger.info("WAV de trabalho já existe e é válido, reutilizando: %s", dest)
        return read_wav_info(dest)

    if duration:
        need = estimate_wav_bytes(duration) * (2 if reserve_for_chunks else 1) + 300 * 1024 ** 2
        ensure_disk_space(dest.parent, need, purpose="converter o áudio")

    partial = dest.with_name(dest.stem + ".partial.wav")
    if partial.exists():
        partial.unlink()

    args = [
        ffmpeg.ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error",
        "-progress", "pipe:1",
        "-y",
        "-i", str(source),
        "-map", "0:a:0",
        "-vn", "-sn", "-dn",
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(partial),
    ]

    def _on_stdout(line: str) -> None:
        # -progress emite "out_time_us=123456" (o campo "out_time_ms" também é em µs).
        if not progress_cb or "=" not in line:
            return
        key, _, value = line.partition("=")
        if key in ("out_time_us", "out_time_ms") and value.strip().lstrip("-").isdigit():
            seconds = max(0.0, int(value) / 1_000_000)
            fraction = min(1.0, seconds / duration) if duration else None
            progress_cb(fraction, seconds)

    try:
        result = run_process(
            args,
            on_stdout_line=_on_stdout,
            cancel_check=cancel_check,
            on_start=on_process_start,
            on_exit=on_process_exit,
            idle_timeout=900,
        )
    except BaseException:  # cancelamento, timeout, erro ao iniciar: não deixa WAV parcial
        _silent_unlink(partial)
        raise
    if result.returncode != 0:
        _silent_unlink(partial)
        raise _ffmpeg_error(result.stderr_text, "Falha ao converter o áudio com o FFmpeg.")

    info = read_wav_info(partial)
    if info.frames == 0:
        _silent_unlink(partial)
        raise CorruptedMediaError("O áudio extraído está vazio.")
    os.replace(partial, dest)
    if progress_cb:
        progress_cb(1.0, info.duration)
    return read_wav_info(dest)


def _silent_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


# =============================================================================
# Detecção de silêncio
# =============================================================================


@dataclass
class SilenceSettings:
    """Parâmetros ajustáveis da detecção de pausas."""

    noise_db: float = DEFAULT_SILENCE_NOISE_DB
    min_duration: float = DEFAULT_SILENCE_MIN_DURATION
    search_window: float = DEFAULT_SEARCH_WINDOW

    @property
    def filter_string(self) -> str:
        return f"silencedetect=noise={self.noise_db:g}dB:d={self.min_duration:g}"

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "SilenceSettings":
        data = data or {}
        return cls(
            noise_db=float(data.get("noise_db", DEFAULT_SILENCE_NOISE_DB)),
            min_duration=float(data.get("min_duration", DEFAULT_SILENCE_MIN_DURATION)),
            search_window=float(data.get("search_window", DEFAULT_SEARCH_WINDOW)),
        )


@dataclass(frozen=True)
class SilenceInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


_SIL_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_silencedetect_output(lines: list[str], offset: float = 0.0, window_end: Optional[float] = None) -> list[SilenceInterval]:
    """Converte a saída do ``silencedetect`` em intervalos absolutos.

    Com ``-ss`` antes de ``-i`` os tempos reportados começam em 0; ``offset``
    (o início da janela) converte para o tempo absoluto do áudio. Um silêncio
    ainda aberto no fim da janela termina em ``window_end``.
    """
    intervals: list[SilenceInterval] = []
    current: Optional[float] = None
    for line in lines:
        m = _SIL_START_RE.search(line)
        if m:
            current = max(0.0, float(m.group(1))) + offset
            continue
        m = _SIL_END_RE.search(line)
        if m:
            end = float(m.group(1)) + offset
            start = current if current is not None else offset
            if end > start:
                intervals.append(SilenceInterval(start, end))
            current = None
    if current is not None and window_end is not None and window_end > current:
        intervals.append(SilenceInterval(current, window_end))
    return intervals


def detect_silences(
    wav_path: Path | str,
    start: float,
    end: float,
    *,
    settings: Optional[SilenceSettings] = None,
    ffmpeg: Optional[FFmpegInfo] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> list[SilenceInterval]:
    """Executa ``silencedetect`` apenas na janela ``[start, end]`` (segundos)."""
    settings = settings or SilenceSettings()
    ffmpeg = ffmpeg or find_ffmpeg()
    if end <= start:
        return []
    lines: list[str] = []
    result = run_process(
        [
            ffmpeg.ffmpeg, "-hide_banner", "-nostdin", "-nostats",
            "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
            "-i", str(wav_path),
            "-af", settings.filter_string,
            "-f", "null", "-",
        ],
        on_stderr_line=lambda ln: lines.append(ln) if "silence_" in ln else None,
        cancel_check=cancel_check,
        timeout=600,
    )
    if result.returncode != 0:
        raise _ffmpeg_error(result.stderr_text, "Falha ao detectar pausas no áudio.")
    return parse_silencedetect_output(lines, offset=start, window_end=end)


# =============================================================================
# Planejamento dos blocos
# =============================================================================


@dataclass
class ChunkSpec:
    """Bloco de áudio definido por amostras inteiras (limites exatos)."""

    index: int
    start_sample: int
    end_sample: int
    sample_rate: int = TARGET_SAMPLE_RATE
    cut_reason: str = "nominal"  # "silence" | "nominal" | "end"
    path: Optional[str] = None

    @property
    def start(self) -> float:
        return self.start_sample / float(self.sample_rate)

    @property
    def end(self) -> float:
        return self.end_sample / float(self.sample_rate)

    @property
    def duration(self) -> float:
        return (self.end_sample - self.start_sample) / float(self.sample_rate)

    @property
    def frames(self) -> int:
        return self.end_sample - self.start_sample

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["start"] = round(self.start, 6)
        d["end"] = round(self.end, 6)
        d["duration"] = round(self.duration, 6)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkSpec":
        return cls(
            index=int(data["index"]),
            start_sample=int(data["start_sample"]),
            end_sample=int(data["end_sample"]),
            sample_rate=int(data.get("sample_rate", TARGET_SAMPLE_RATE)),
            cut_reason=str(data.get("cut_reason", "nominal")),
            path=data.get("path"),
        )


def choose_cut_point(
    silences: list[SilenceInterval],
    target: float,
    lo: float,
    hi: float,
    *,
    long_pause_bonus: float = 5.0,
    bonus_cap: float = 2.0,
) -> Optional[float]:
    """Escolhe o ponto de corte (segundos) dentro de ``[lo, hi]`` ou ``None``.

    Cada pausa é recortada à janela; o corte é o MEIO da pausa (longe das
    palavras vizinhas). Pontuação = distância até o alvo − bônus por pausa
    longa (pausas maiores tendem a separar frases). Ex.: uma pausa de 2 s a
    15 s do alvo vence uma de 0,4 s a 7 s.
    """
    best: Optional[tuple[float, float]] = None  # (score, cut)
    for s in silences:
        a, b = max(s.start, lo), min(s.end, hi)
        if b <= a:
            continue
        mid = (a + b) / 2.0
        score = abs(mid - target) - long_pause_bonus * min(b - a, bonus_cap)
        if best is None or score < best[0]:
            best = (score, mid)
    return best[1] if best else None


SilenceFinder = Callable[[float, float], list[SilenceInterval]]


def plan_chunks(
    total_frames: int,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    search_window: float = DEFAULT_SEARCH_WINDOW,
    silence_finder: Optional[SilenceFinder] = None,
    cancel_check: Optional[CancelCheck] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> list[ChunkSpec]:
    """Planeja os blocos com cortes em pausas próximas de cada limite nominal.

    Garantias (verificadas por ``verify_chunk_continuity``):
      * o 1º bloco começa na amostra 0 e o último termina em ``total_frames``;
      * ``fim(N) == início(N+1)`` — sem lacunas nem sobreposições;
      * todo bloco tem duração positiva.

    O próximo limite nominal é contado a partir do corte REAL anterior, então
    os blocos ficam sempre próximos de ``chunk_seconds``.
    """
    if total_frames <= 0:
        raise AudioProcessingError("O áudio está vazio.")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds deve ser positivo")
    search_window = max(0.0, min(search_window, chunk_seconds / 2.0))

    chunk = int(round(chunk_seconds * sample_rate))
    window = int(round(search_window * sample_rate))
    min_chunk = max(1, chunk - window)
    expected_total = max(1, -(-total_frames // chunk))  # divisão para cima

    chunks: list[ChunkSpec] = []
    pos = 0
    while True:
        if cancel_check and cancel_check():
            raise TaskCancelledError()
        remaining = total_frames - pos
        # Último bloco: o restante cabe em um bloco nominal + janela.
        if remaining <= chunk + window:
            chunks.append(ChunkSpec(len(chunks), pos, total_frames, sample_rate, "end"))
            break

        target = pos + chunk
        lo = pos + min_chunk
        hi = min(target + window, total_frames - 1)
        cut: Optional[int] = None
        reason = "nominal"
        if silence_finder is not None and window > 0:
            silences = silence_finder(lo / sample_rate, hi / sample_rate)
            point = choose_cut_point(silences, target / sample_rate, lo / sample_rate, hi / sample_rate)
            if point is not None:
                cut = int(round(point * sample_rate))
                cut = min(max(cut, lo), hi)
                reason = "silence"
        if cut is None:
            cut = target
        if cut <= pos:  # proteção extra (não deveria acontecer)
            cut = min(total_frames, pos + chunk)
        chunks.append(ChunkSpec(len(chunks), pos, cut, sample_rate, reason))
        pos = cut
        if progress_cb:
            progress_cb(len(chunks), max(expected_total, len(chunks) + 1))

    if progress_cb:
        progress_cb(len(chunks), len(chunks))
    verify_chunk_continuity(chunks, total_frames)
    return chunks


def verify_chunk_continuity(chunks: list[ChunkSpec], total_frames: int) -> None:
    """Lança ``AudioProcessingError`` se houver lacuna, sobreposição ou bloco vazio."""
    if not chunks:
        raise AudioProcessingError("Nenhum bloco foi planejado.")
    if chunks[0].start_sample != 0:
        raise AudioProcessingError(technical=f"1º bloco começa em {chunks[0].start_sample}")
    for i, c in enumerate(chunks):
        if c.index != i:
            raise AudioProcessingError(technical=f"Índice fora de ordem: {c.index} != {i}")
        if c.end_sample <= c.start_sample:
            raise AudioProcessingError(technical=f"Bloco {i} vazio ou invertido")
        if i + 1 < len(chunks) and chunks[i + 1].start_sample != c.end_sample:
            raise AudioProcessingError(
                technical=f"Descontinuidade entre blocos {i} e {i + 1}: {c.end_sample} != {chunks[i + 1].start_sample}"
            )
    if chunks[-1].end_sample != total_frames:
        raise AudioProcessingError(technical=f"Último bloco termina em {chunks[-1].end_sample} != {total_frames}")


def build_chunk_plan(
    wav_path: Path | str,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    settings: Optional[SilenceSettings] = None,
    ffmpeg: Optional[FFmpegInfo] = None,
    use_silence: bool = True,
    cancel_check: Optional[CancelCheck] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> list[ChunkSpec]:
    """Lê o WAV de trabalho e planeja os blocos usando ``silencedetect`` nas janelas."""
    settings = settings or SilenceSettings()
    info = read_wav_info(wav_path)
    finder: Optional[SilenceFinder] = None
    if use_silence:
        ff = ffmpeg or find_ffmpeg()

        def finder(a: float, b: float) -> list[SilenceInterval]:
            try:
                return detect_silences(wav_path, a, b, settings=settings, ffmpeg=ff, cancel_check=cancel_check)
            except TaskCancelledError:
                raise
            except TranscriberError as exc:
                # Falha pontual: usa o limite nominal (continuidade preservada).
                logger.warning("silencedetect falhou na janela %.1f–%.1f s: %s", a, b, exc)
                return []

    return plan_chunks(
        info.frames,
        sample_rate=info.sample_rate,
        chunk_seconds=chunk_seconds,
        search_window=settings.search_window,
        silence_finder=finder,
        cancel_check=cancel_check,
        progress_cb=progress_cb,
    )


# =============================================================================
# Gravação dos blocos
# =============================================================================

_COPY_FRAMES = TARGET_SAMPLE_RATE * 5  # copia 5 s por vez (~160 KB)


def chunk_filename(index: int) -> str:
    return f"chunk_{index:04d}.wav"


def export_chunk(
    wav_path: Path | str,
    chunk: ChunkSpec,
    out_path: Path | str,
    *,
    cancel_check: Optional[CancelCheck] = None,
) -> Path:
    """Copia as amostras ``[start_sample, end_sample)`` para um WAV próprio (gravação atômica)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.wav")
    try:
        with wave.open(str(wav_path), "rb") as src:
            if src.getnframes() < chunk.end_sample:
                raise AudioProcessingError(technical=f"Bloco {chunk.index} ultrapassa o fim do WAV")
            src.setpos(chunk.start_sample)
            with wave.open(str(tmp), "wb") as dst:
                dst.setnchannels(src.getnchannels())
                dst.setsampwidth(src.getsampwidth())
                dst.setframerate(src.getframerate())
                remaining = chunk.frames
                while remaining > 0:
                    if cancel_check and cancel_check():
                        raise TaskCancelledError()
                    n = min(_COPY_FRAMES, remaining)
                    data = src.readframes(n)
                    if not data:
                        raise AudioProcessingError(technical=f"Fim inesperado do WAV no bloco {chunk.index}")
                    dst.writeframes(data)
                    remaining -= len(data) // (src.getsampwidth() * src.getnchannels())
        os.replace(tmp, out)
    except BaseException:
        _silent_unlink(tmp)
        raise
    return out


def _chunk_file_ok(path: Path, chunk: ChunkSpec) -> bool:
    try:
        return path.is_file() and read_wav_info(path).frames == chunk.frames
    except AudioProcessingError:
        return False


def split_wav(
    wav_path: Path | str,
    chunks: list[ChunkSpec],
    out_dir: Path | str,
    *,
    cancel_check: Optional[CancelCheck] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> list[ChunkSpec]:
    """Grava todos os blocos em ``out_dir`` (reaproveita os já gravados corretamente).

    Retorna os ``ChunkSpec`` com ``path`` preenchido. Uso de RAM constante
    (~160 KB de buffer), independentemente da duração.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(c.frames for c in chunks) * TARGET_SAMPLE_WIDTH * TARGET_CHANNELS
    already = sum(
        c.frames for c in chunks if _chunk_file_ok(out_dir / chunk_filename(c.index), c)
    ) * TARGET_SAMPLE_WIDTH
    ensure_disk_space(out_dir, max(0, total_bytes - already) + 100 * 1024 ** 2, purpose="dividir o áudio em blocos")

    result: list[ChunkSpec] = []
    for i, chunk in enumerate(chunks):
        path = out_dir / chunk_filename(chunk.index)
        if not _chunk_file_ok(path, chunk):
            export_chunk(wav_path, chunk, path, cancel_check=cancel_check)
        result.append(ChunkSpec(chunk.index, chunk.start_sample, chunk.end_sample, chunk.sample_rate,
                                chunk.cut_reason, str(path)))
        if progress_cb:
            progress_cb(i + 1, len(chunks))
    return result


def chunk_offset_seconds(chunk: ChunkSpec) -> float:
    """Offset a somar aos timestamps do bloco para obter o tempo do vídeo original."""
    return chunk.start


def remove_chunk_files(chunks: list[ChunkSpec]) -> None:
    """Remove arquivos temporários de blocos (nunca toca no arquivo original)."""
    for c in chunks:
        if c.path:
            _silent_unlink(Path(c.path))
