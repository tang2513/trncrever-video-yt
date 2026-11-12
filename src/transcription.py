"""Motor de transcrição: faster-whisper aplicado a UM bloco de áudio por vez.

Responsabilidades:

* Montar os parâmetros do faster-whisper (VAD, beam size, limiares contra
  alucinações/repetições) a partir de ``TranscriptionSettings``.
* Carregar o modelo no dispositivo escolhido por ``hardware.resolve_device``,
  com novas tentativas para falhas de rede (download do modelo no 1º uso),
  verificação de RAM e *fallback* para CPU quando a GPU falha.
* Transcrever um bloco (WAV de ~10 min no disco) consumindo os segmentos em
  fluxo, convertendo os tempos do bloco para o TEMPO ORIGINAL do vídeo
  (``offset`` = início do bloco) e permitindo cancelar entre segmentos.

Decisões que reduzem alucinações sem alterar o conteúdo falado:

* ``condition_on_previous_text=False``: evita que um erro/repetição se
  propague para o texto seguinte (causa clássica de laços de repetição).
* ``vad_filter=True`` com ``speech_pad_ms`` generoso: remove trechos sem fala
  (onde o Whisper "inventa" texto) sem cortar o início/fim das palavras.
* Fallback de temperatura + ``compression_ratio_threshold`` e
  ``log_prob_threshold``: refaz trechos com saída repetitiva ou pouco confiável.
* Nenhuma deduplicação textual é feita: se o palestrante repetiu uma frase,
  ela permanece.

O módulo não importa o faster-whisper no topo: o import acontece sob demanda
(depois de registrar as DLLs CUDA), o que mantém a interface leve e permite
testar a lógica com um ``model_factory`` falso.
"""

from __future__ import annotations

import errno
import gc
import logging
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import MODELS_DIR
from .hardware import (
    DeviceDecision,
    cpu_fallback_decision,
    is_cuda_runtime_error,
    register_cuda_dll_dirs,
)
from .subtitles import Segment
from .utils import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    MODELS,
    DiskSpaceError,
    GPUError,
    InsufficientMemoryError,
    ModelLoadError,
    NetworkError,
    TaskCancelledError,
    TranscriberError,
    format_bytes,
    get_available_ram,
    retry_call,
)

logger = logging.getLogger(__name__)

GB = 1024 ** 3

CancelCheck = Callable[[], bool]
#: ``progress_cb(segundos_processados_no_bloco, duração_do_bloco)``
BlockProgress = Callable[[float, float], None]
MessageCallback = Callable[[str], None]
DeviceChangeCallback = Callable[[DeviceDecision, str], None]


# =============================================================================
# Configuração
# =============================================================================


@dataclass
class TranscriptionSettings:
    """Parâmetros de transcrição (serializáveis no ``config.json`` da tarefa)."""

    model: str = DEFAULT_MODEL
    #: Código do Whisper ("pt", "en"...) ou ``None`` para detecção automática.
    language: Optional[str] = DEFAULT_LANGUAGE
    beam_size: int = 5
    best_of: int = 5
    vad_filter: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 400
    condition_on_previous_text: bool = False
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    initial_prompt: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["temperature"] = list(self.temperature)
        return data

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "TranscriptionSettings":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        if "temperature" in clean and not isinstance(clean["temperature"], (int, float)):
            clean["temperature"] = tuple(float(t) for t in clean["temperature"])
        return cls(**clean)


def normalize_language(code: Optional[str]) -> Optional[str]:
    """``"auto"``/vazio → ``None`` (detecção automática); caso contrário o código em minúsculas."""
    value = (code or "").strip().lower()
    return None if value in ("", "auto") else value


def build_transcribe_kwargs(settings: TranscriptionSettings, language: Optional[str]) -> dict[str, Any]:
    """Argumentos de ``WhisperModel.transcribe`` (separado para ser testável)."""
    kwargs: dict[str, Any] = {
        "language": normalize_language(language),
        "task": "transcribe",
        "beam_size": max(1, int(settings.beam_size)),
        "best_of": max(1, int(settings.best_of)),
        "temperature": list(settings.temperature),
        "compression_ratio_threshold": settings.compression_ratio_threshold,
        "log_prob_threshold": settings.log_prob_threshold,
        "no_speech_threshold": settings.no_speech_threshold,
        "condition_on_previous_text": settings.condition_on_previous_text,
        "initial_prompt": settings.initial_prompt,
        "vad_filter": bool(settings.vad_filter),
        "word_timestamps": False,
    }
    if settings.vad_filter:
        kwargs["vad_parameters"] = {
            "threshold": settings.vad_threshold,
            "min_silence_duration_ms": settings.vad_min_silence_ms,
            "speech_pad_ms": settings.vad_speech_pad_ms,
        }
    return kwargs


# =============================================================================
# Conversão de segmentos / timestamps
# =============================================================================


def make_segment(start: float, end: float, text: str, offset: float, chunk_duration: float) -> Optional[Segment]:
    """Cria um ``Segment`` com tempo ORIGINAL do vídeo a partir de tempos do bloco.

    * Texto vazio → ``None``.
    * Os tempos são limitados a ``[0, chunk_duration]``: nenhuma fala "vaza"
      para o bloco vizinho, preservando a continuidade exata entre blocos.
    """
    text = (text or "").strip()
    if not text:
        return None
    s = min(max(float(start), 0.0), chunk_duration)
    e = min(max(float(end), s), chunk_duration)
    return Segment(offset + s, offset + e, text)


def convert_raw_segments(raw: Iterable[Any], offset: float, chunk_duration: float) -> list[Segment]:
    """Converte objetos com ``start``/``end``/``text`` (faster-whisper) em ``Segment``."""
    out: list[Segment] = []
    for r in raw:
        seg = make_segment(r.start, r.end, r.text, offset, chunk_duration)
        if seg is not None:
            out.append(seg)
    return out


@dataclass
class ChunkTranscription:
    segments: list[Segment]
    language: Optional[str]
    language_probability: Optional[float]
    chunk_duration: float
    elapsed: float

    @property
    def speech_detected(self) -> bool:
        return bool(self.segments)

    @property
    def realtime_factor(self) -> Optional[float]:
        """Segundos de processamento por segundo de áudio (menor é mais rápido)."""
        return self.elapsed / self.chunk_duration if self.chunk_duration > 0 else None


# =============================================================================
# Erros de carregamento
# =============================================================================

_NETWORK_KEYS = (
    "connectionerror", "connection error", "connection reset", "connection aborted", "timed out", "timeout",
    "name resolution", "getaddrinfo", "max retries", "offline", "ssl", "proxy", "temporary failure",
    "couldn't connect", "cannot connect", "localentrynotfounderror", "locate the files on the hub",
    "network is unreachable", "remote end closed",
)


def looks_like_network_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(key in text for key in _NETWORK_KEYS)


def _default_model_factory(
    model_name: str,
    device: str,
    device_index: int,
    compute_type: str,
    cpu_threads: int,
    download_root: str,
) -> Any:
    """Cria o ``WhisperModel`` (import tardio, depois de registrar as DLLs CUDA)."""
    if device == "cuda":
        register_cuda_dll_dirs()
    from faster_whisper import WhisperModel  # noqa: WPS433

    return WhisperModel(
        model_name,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
        download_root=download_root,
    )


ModelFactory = Callable[[str, str, int, str, int, str], Any]


# =============================================================================
# Motor
# =============================================================================


class TranscriptionEngine:
    """Mantém o modelo carregado e transcreve blocos, um de cada vez.

    ``allow_cpu_fallback``: se ``True``, uma falha de GPU (ao carregar ou
    durante a transcrição) faz o motor recarregar o modelo na CPU e continuar.
    Se ``False`` (usuário escolheu GPU explicitamente), lança ``GPUError`` para
    que a interface ofereça a CPU.
    """

    def __init__(
        self,
        settings: TranscriptionSettings,
        decision: DeviceDecision,
        *,
        model_factory: Optional[ModelFactory] = None,
        download_root: Path | str = MODELS_DIR,
        allow_cpu_fallback: bool = True,
        on_message: Optional[MessageCallback] = None,
        on_device_change: Optional[DeviceChangeCallback] = None,
    ) -> None:
        if settings.model not in MODELS:
            raise ValueError(f"Modelo desconhecido: {settings.model}")
        self.settings = settings
        self.decision = decision
        self.allow_cpu_fallback = allow_cpu_fallback
        self._factory: ModelFactory = model_factory or _default_model_factory
        self._download_root = str(download_root)
        self._on_message = on_message
        self._on_device_change = on_device_change
        self._model: Any = None

    # -- utilidades ----------------------------------------------------------

    def _say(self, text: str) -> None:
        logger.info(text)
        if self._on_message:
            self._on_message(text)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- carregamento --------------------------------------------------------

    def _check_memory(self, decision: DeviceDecision) -> None:
        info = MODELS.get(self.settings.model)
        if info is None or decision.device != "cpu":
            return
        need = info.ram_gb_int8 * GB if decision.compute_type.startswith("int8") else info.ram_gb_int8 * 2.2 * GB
        available = get_available_ram()
        if available < need * 0.9:
            raise InsufficientMemoryError(
                f"Memória RAM insuficiente para o modelo '{self.settings.model}': são necessários cerca de "
                f"{format_bytes(need)} e há {format_bytes(available)} livres. Feche outros programas ou "
                f"escolha um modelo menor.",
                technical=f"available={available} need={need}",
            )

    def _load_once(self, decision: DeviceDecision) -> Any:
        try:
            return self._factory(
                self.settings.model,
                decision.device,
                decision.device_index,
                decision.compute_type,
                decision.cpu_threads,
                self._download_root,
            )
        except TaskCancelledError:
            raise
        except MemoryError as exc:
            raise InsufficientMemoryError(technical=repr(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - qualquer falha do backend vira erro amigável
            if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
                raise DiskSpaceError(technical=repr(exc)) from exc
            if looks_like_network_error(exc):
                raise NetworkError(technical=repr(exc)) from exc
            raise ModelLoadError(technical=repr(exc)) from exc

    def load(self, cancel_check: Optional[CancelCheck] = None) -> None:
        """Carrega o modelo (idempotente). Pode lançar ``ModelLoadError``, ``GPUError`` etc."""
        if self._model is not None:
            return
        self._check_memory(self.decision)
        self._say(
            f"Carregando o modelo '{self.settings.model}' em {self.decision.label}. "
            "No primeiro uso o modelo é baixado da internet (pode demorar)."
        )

        def _on_retry(attempt: int, delay: float, exc: BaseException) -> None:
            self._say(f"Falha de conexão ao baixar o modelo (tentativa {attempt}). Nova tentativa em {delay:.0f}s…")

        try:
            self._model = retry_call(
                lambda: self._load_once(self.decision),
                attempts=3,
                base_delay=4.0,
                retry_on=(NetworkError,),
                cancel_check=cancel_check,
                on_retry=_on_retry,
            )
        except NetworkError as exc:
            raise ModelLoadError(technical=exc.technical) from exc
        except ModelLoadError as exc:
            # Só falhas do próprio carregamento (ex.: CUDA/cuDNN) justificam trocar para a CPU;
            # falta de RAM ou de disco continuam sendo erros próprios.
            if self.decision.device == "cuda":
                self._handle_gpu_failure(exc.technical or str(exc), cancel_check)
            else:
                raise
        self._say(f"Modelo '{self.settings.model}' carregado ({self.decision.label}).")

    def _handle_gpu_failure(self, reason: str, cancel_check: Optional[CancelCheck] = None) -> None:
        """Recarrega o modelo na CPU ou lança ``GPUError`` (conforme ``allow_cpu_fallback``)."""
        if not self.allow_cpu_fallback:
            raise GPUError(
                "A GPU apresentou um erro durante o processamento. Você pode continuar pela CPU.",
                technical=reason,
            )
        self.close()
        new_decision = cpu_fallback_decision(self.settings.model, reason[:200])
        self._check_memory(new_decision)
        self._say("A GPU falhou; continuando o processamento pela CPU (mais lento).")
        self.decision = new_decision
        self._model = self._load_once(new_decision)
        if self._on_device_change:
            self._on_device_change(new_decision, reason)

    def close(self) -> None:
        """Libera o modelo da memória."""
        self._model = None
        gc.collect()

    # -- transcrição ---------------------------------------------------------

    def _transcribe_once(
        self,
        wav_path: Path | str,
        offset: float,
        chunk_duration: float,
        language: Optional[str],
        progress_cb: Optional[BlockProgress],
        cancel_check: Optional[CancelCheck],
    ) -> ChunkTranscription:
        started = time.monotonic()
        kwargs = build_transcribe_kwargs(self.settings, language)
        segments_iter, info = self._model.transcribe(str(wav_path), **kwargs)

        segments: list[Segment] = []
        # O gerador só processa o áudio quando é consumido: cancelar entre
        # segmentos interrompe o trabalho em um ponto seguro.
        for raw in segments_iter:
            if cancel_check and cancel_check():
                raise TaskCancelledError()
            seg = make_segment(raw.start, raw.end, raw.text, offset, chunk_duration)
            if seg is not None:
                segments.append(seg)
            if progress_cb:
                progress_cb(min(max(float(raw.end), 0.0), chunk_duration), chunk_duration)

        if cancel_check and cancel_check():
            raise TaskCancelledError()
        if progress_cb:
            progress_cb(chunk_duration, chunk_duration)

        lang = getattr(info, "language", None)
        prob = getattr(info, "language_probability", None)
        return ChunkTranscription(
            segments=segments,
            language=lang,
            language_probability=float(prob) if prob is not None else None,
            chunk_duration=chunk_duration,
            elapsed=time.monotonic() - started,
        )

    def transcribe_file(
        self,
        wav_path: Path | str,
        *,
        offset: float,
        chunk_duration: float,
        language: Optional[str] = None,
        progress_cb: Optional[BlockProgress] = None,
        cancel_check: Optional[CancelCheck] = None,
    ) -> ChunkTranscription:
        """Transcreve um bloco. Os tempos retornados já estão no tempo original do vídeo."""
        if self._model is None:
            self.load(cancel_check)
        try:
            return self._transcribe_once(wav_path, offset, chunk_duration, language, progress_cb, cancel_check)
        except TaskCancelledError:
            raise
        except MemoryError as exc:
            raise InsufficientMemoryError(technical=repr(exc)) from exc
        except TranscriberError:
            raise
        except Exception as exc:  # noqa: BLE001
            if self.decision.device == "cuda" and is_cuda_runtime_error(exc):
                logger.warning("Falha de CUDA durante a transcrição: %r", exc)
                self._handle_gpu_failure(repr(exc), cancel_check)
                return self._transcribe_once(wav_path, offset, chunk_duration, language, progress_cb, cancel_check)
            raise TranscriberError(
                "Ocorreu uma falha ao transcrever um trecho do áudio.", technical=repr(exc)
            ) from exc
