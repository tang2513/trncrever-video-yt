"""Links do YouTube e da Twitch: validação, metadados, legendas e áudio (via yt-dlp).

Responsabilidades:

* Detectar o tipo de entrada (arquivo local, YouTube, Twitch) e normalizar links.
* Verificar se o ``yt-dlp`` está instalado.
* Obter metadados (título, duração, legendas disponíveis) sem baixar mídia.
* Baixar somente a legenda escolhida (sem áudio), quando houver legenda adequada.
* Baixar somente o ÁUDIO (``bestaudio``); se não existir faixa só de áudio,
  baixa o menor formato que contenha áudio — nunca o vídeo em alta resolução.
* Classificar erros do yt-dlp em mensagens claras (indisponível, expirado,
  privado, exige login, protegido por DRM/assinatura, falha de rede).
* Novas tentativas com intervalos progressivos para falhas de rede.
* Cancelamento: o yt-dlp roda como processo filho e é encerrado (com a árvore
  de processos) quando o cancelamento é solicitado.

Este módulo NÃO contorna DRM, paywalls, conteúdo para membros ou restrições de
acesso. Cookies são aceitos apenas de uma sessão autorizada do próprio usuário,
e o arquivo original de cookies nunca é modificado (o yt-dlp trabalha com uma
cópia temporária dentro da pasta da tarefa).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from .subtitles import SubtitleTrack
from .utils import (
    AuthenticationRequiredError,
    ContentUnavailableError,
    DownloadError,
    InvalidInputError,
    NetworkError,
    ProtectedContentError,
    SubtitleError,
    TranscriberError,
    clean_user_path,
    ensure_disk_space,
    retry_call,
    run_process,
    subprocess_env,
)

logger = logging.getLogger(__name__)

SOURCE_LOCAL = "local"
SOURCE_YOUTUBE = "youtube"
SOURCE_TWITCH = "twitch"

SOURCE_LABELS = {SOURCE_LOCAL: "Arquivo local", SOURCE_YOUTUBE: "YouTube", SOURCE_TWITCH: "Twitch"}

#: Seleção de formato: só áudio; senão o menor formato que tenha áudio.
AUDIO_FORMAT_SELECTOR = "bestaudio/worst[acodec!=none]/worst"

_YT_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com",
}
_TWITCH_HOSTS = {"twitch.tv", "www.twitch.tv", "m.twitch.tv", "go.twitch.tv"}
_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

ProgressCallback = Callable[["DownloadProgress"], None]
CancelCheck = Callable[[], bool]
RetryCallback = Callable[[int, float, BaseException], None]


# =============================================================================
# Validação de links
# =============================================================================


@dataclass(frozen=True)
class ValidatedURL:
    source_type: str  # "youtube" ou "twitch"
    url: str          # URL canônica
    media_id: str


def looks_like_url(text: str) -> bool:
    t = (text or "").strip().lower()
    return t.startswith(("http://", "https://", "www.")) or any(
        t.startswith(h) for h in ("youtube.com", "youtu.be", "m.youtube.com", "twitch.tv", "m.twitch.tv")
    )


def _parse(url: str):
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return urlparse(url)


def normalize_youtube_url(url: str) -> ValidatedURL:
    parsed = _parse(url)
    host = (parsed.hostname or "").lower()
    if host not in _YT_HOSTS:
        raise InvalidInputError("O link informado não é do YouTube.")
    path = parsed.path or "/"
    video_id: Optional[str] = None
    if host.endswith("youtu.be"):
        video_id = path.strip("/").split("/")[0]
    elif path == "/watch":
        video_id = (parse_qs(parsed.query).get("v") or [None])[0]
    else:
        m = re.match(r"^/(?:live|shorts|embed|v|e)/([^/?#]+)", path)
        if m:
            video_id = m.group(1)
    if not video_id:
        if "list" in parse_qs(parsed.query) or path.startswith("/playlist"):
            raise InvalidInputError("Links de playlist não são suportados. Cole o link de um vídeo específico.")
        if path.startswith(("/@", "/channel", "/c/", "/user")):
            raise InvalidInputError("Este é o link de um canal. Cole o link de um vídeo específico.")
        raise InvalidInputError("Não foi possível identificar o vídeo neste link do YouTube.")
    if not _YT_ID_RE.match(video_id):
        raise InvalidInputError("O identificador do vídeo no link do YouTube é inválido.")
    return ValidatedURL(SOURCE_YOUTUBE, f"https://www.youtube.com/watch?v={video_id}", video_id)


def normalize_twitch_url(url: str) -> ValidatedURL:
    parsed = _parse(url)
    host = (parsed.hostname or "").lower()
    if host == "clips.twitch.tv" or "/clip/" in (parsed.path or ""):
        raise InvalidInputError("Clipes não são suportados. Cole o link de uma gravação (VOD) da Twitch.")
    if host not in _TWITCH_HOSTS:
        raise InvalidInputError("O link informado não é da Twitch.")
    path = parsed.path or "/"
    m = re.match(r"^/videos/(\d+)", path) or re.match(r"^/[^/]+/(?:v|video)/(\d+)", path)
    if not m:
        if re.match(r"^/[^/]+/?$", path):
            raise InvalidInputError(
                "Este é o link de um canal ou transmissão ao vivo. Cole o link de uma gravação (VOD), "
                "no formato https://www.twitch.tv/videos/123456789."
            )
        raise InvalidInputError("Não foi possível identificar o VOD neste link da Twitch.")
    vod_id = m.group(1)
    return ValidatedURL(SOURCE_TWITCH, f"https://www.twitch.tv/videos/{vod_id}", vod_id)


def validate_url(url: str) -> ValidatedURL:
    """Valida e normaliza um link do YouTube ou da Twitch."""
    text = clean_user_path(url)
    if not text:
        raise InvalidInputError("Cole o link do vídeo.")
    host = (_parse(text).hostname or "").lower()
    if host in _YT_HOSTS:
        return normalize_youtube_url(text)
    if host in _TWITCH_HOSTS or host == "clips.twitch.tv":
        return normalize_twitch_url(text)
    raise InvalidInputError("Link não suportado. Use um link do YouTube ou de um VOD da Twitch.")


def detect_source_type(text: str) -> str:
    """``"youtube"``, ``"twitch"`` ou ``"local"`` (não valida o conteúdo)."""
    if looks_like_url(text):
        host = (_parse(clean_user_path(text)).hostname or "").lower()
        if host in _YT_HOSTS:
            return SOURCE_YOUTUBE
        if host in _TWITCH_HOSTS or host == "clips.twitch.tv":
            return SOURCE_TWITCH
    return SOURCE_LOCAL


# =============================================================================
# yt-dlp
# =============================================================================


def check_ytdlp() -> str:
    """Retorna a versão do yt-dlp ou lança ``TranscriberError`` se ausente."""
    try:
        result = run_process([sys.executable, "-m", "yt_dlp", "--version"], timeout=60)
    except TranscriberError as exc:
        raise TranscriberError(
            "O yt-dlp não está instalado. Execute: python -m pip install -U yt-dlp", technical=exc.technical
        ) from exc
    if result.returncode != 0 or not result.stdout_tail:
        raise TranscriberError(
            "O yt-dlp não está instalado. Execute: python -m pip install -U yt-dlp",
            technical=result.stderr_text[-1000:],
        )
    return result.stdout_tail[-1].strip()


def prepare_cookies_copy(cookies_file: Optional[str | Path], work_dir: Path) -> Optional[Path]:
    """Copia o cookies.txt do usuário para a pasta da tarefa.

    O yt-dlp REESCREVE o arquivo de cookies informado; trabalhar com uma cópia
    preserva o original. A cópia deve ser apagada ao final da tarefa.
    """
    if not cookies_file:
        return None
    src = Path(clean_user_path(str(cookies_file)))
    if not src.is_file():
        raise InvalidInputError(f"Arquivo de cookies não encontrado: {src}")
    head = src.read_text(encoding="utf-8", errors="ignore")[:4096]
    if "\t" not in head and "# Netscape" not in head:
        raise InvalidInputError(
            "O arquivo de cookies precisa estar no formato Netscape (cookies.txt), exportado de uma sessão sua."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    dst = work_dir / "cookies.txt"
    shutil.copyfile(src, dst)
    return dst


def _base_args(cookies: Optional[Path], ffmpeg_dir: Optional[str] = None) -> list[str]:
    args = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--ignore-config",
        "--retries", "10",
        "--fragment-retries", "10",
        "--socket-timeout", "30",
        "--encoding", "utf-8",
    ]
    if cookies:
        args += ["--cookies", str(cookies)]
    if ffmpeg_dir:
        args += ["--ffmpeg-location", ffmpeg_dir]
    return args


_ERROR_RULES: list[tuple[tuple[str, ...], type[TranscriberError], Optional[str]]] = [
    (("drm",), ProtectedContentError, None),
    (("members-only", "members only", "join this channel", "subscriber-only", "subscribers only",
      "only available for subscribers", "requires payment", "purchase this", "rental"), ProtectedContentError, None),
    (("sign in to confirm your age", "age-restricted", "age restricted", "confirm you're not a bot",
      "confirm you’re not a bot", "login required", "use --cookies", "account cookies",
      "authentication", "requires authentication", "log in"), AuthenticationRequiredError, None),
    (("is currently live", "live event will begin", "premieres in", "is_upcoming", "stream is offline"),
     ContentUnavailableError,
     "A transmissão ainda está ao vivo ou não começou. Aguarde ela terminar e virar gravação (VOD)."),
    (("not available in your country", "geo restriction", "geo-restricted"), ContentUnavailableError,
     "Este vídeo não está disponível na sua região."),
    (("private video", "video unavailable", "has been removed", "does not exist", "http error 404",
      "no longer available", "this video is not available", "vod is unavailable",
      "has been deleted", "account associated with this video has been terminated"),
     ContentUnavailableError, None),
    (("unsupported url",), InvalidInputError, "Este link não é suportado pelo yt-dlp."),
    (("unable to download webpage", "timed out", "timeout", "connection reset", "getaddrinfo",
      "name resolution", "network is unreachable", "http error 5", "http error 429", "too many requests",
      "remote end closed", "ssl", "connection aborted", "connection refused", "incompleteread",
      "unable to download video data", "failed to resolve", "winerror 10054", "winerror 10060"),
     NetworkError, None),
]


def classify_ytdlp_error(stderr: str, *, source_type: str = SOURCE_YOUTUBE) -> TranscriberError:
    """Converte a saída de erro do yt-dlp em uma exceção com mensagem amigável."""
    text = (stderr or "").lower()
    error_lines = [ln for ln in (stderr or "").splitlines() if "ERROR" in ln]
    technical = "\n".join(error_lines[-5:]) or (stderr or "")[-1500:]
    for keys, exc_type, message in _ERROR_RULES:
        if any(k in text for k in keys):
            if exc_type is ContentUnavailableError and message is None and source_type == SOURCE_TWITCH:
                message = "Este VOD da Twitch está indisponível: pode ter expirado, sido removido ou ser privado."
            return exc_type(message, technical=technical)
    return DownloadError(technical=technical)


# =============================================================================
# Metadados
# =============================================================================


@dataclass
class DownloadProgress:
    downloaded_bytes: int = 0
    total_bytes: Optional[int] = None
    fragment_index: Optional[int] = None
    fragment_count: Optional[int] = None

    @property
    def fraction(self) -> Optional[float]:
        """Fração real concluída (0–1) ou ``None`` se não houver como medir."""
        if self.total_bytes and self.total_bytes > 0:
            return max(0.0, min(1.0, self.downloaded_bytes / self.total_bytes))
        if self.fragment_count and self.fragment_index is not None and self.fragment_count > 0:
            return max(0.0, min(1.0, self.fragment_index / self.fragment_count))
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "fragment_index": self.fragment_index,
            "fragment_count": self.fragment_count,
            "fraction": self.fraction,
        }


_KEEP_INFO_KEYS = (
    "id", "title", "fulltitle", "duration", "extractor", "extractor_key", "webpage_url", "uploader",
    "channel", "upload_date", "language", "subtitles", "automatic_captions", "is_live", "was_live",
    "live_status", "availability", "filesize", "filesize_approx", "abr", "tbr", "format_id", "ext",
    "acodec", "vcodec",
)


def fetch_metadata(
    url: str,
    *,
    source_type: str = SOURCE_YOUTUBE,
    cookies: Optional[Path] = None,
    cancel_check: Optional[CancelCheck] = None,
    on_retry: Optional[RetryCallback] = None,
    attempts: int = 4,
) -> dict[str, Any]:
    """Consulta os metadados (``yt-dlp -J``) sem baixar mídia.

    O mesmo seletor de formato do download é usado, para que ``filesize``/
    ``filesize_approx`` correspondam ao áudio que será baixado. Retorna apenas
    os campos necessários (economiza memória).
    """

    def _once() -> dict[str, Any]:
        chunks: list[str] = []
        args = _base_args(cookies) + ["-J", "-f", AUDIO_FORMAT_SELECTOR, url]
        result = run_process(
            args, on_stdout_line=chunks.append, cancel_check=cancel_check, timeout=300, env=subprocess_env()
        )
        if result.returncode != 0:
            raise classify_ytdlp_error(result.stderr_text, source_type=source_type)
        raw = "".join(chunks).strip()
        try:
            info = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DownloadError("Resposta inesperada ao consultar o vídeo.", technical=repr(exc)) from exc
        return {k: info.get(k) for k in _KEEP_INFO_KEYS}

    info = retry_call(_once, attempts=attempts, cancel_check=cancel_check, on_retry=on_retry)

    live_status = (info.get("live_status") or "").lower()
    if info.get("is_live") or live_status in ("is_live", "is_upcoming", "post_live"):
        raise ContentUnavailableError(
            "Este conteúdo ainda está ao vivo ou sendo processado pela plataforma. "
            "Aguarde a gravação ficar disponível e tente novamente."
        )
    availability = (info.get("availability") or "").lower()
    if availability in ("premium_only", "subscriber_only"):
        raise ProtectedContentError()
    return info


def estimate_download_bytes(info: dict[str, Any]) -> int:
    """Estimativa do tamanho do áudio a baixar (para verificar espaço em disco)."""
    for key in ("filesize", "filesize_approx"):
        value = info.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value * 1.1)
    duration = float(info.get("duration") or 0)
    kbps = float(info.get("abr") or info.get("tbr") or 256)
    if duration <= 0:
        return 2 * 1024 ** 3  # desconhecido: exige 2 GB livres por segurança
    return int(duration * kbps * 1000 / 8 * 1.2)


# =============================================================================
# Legendas
# =============================================================================


def download_subtitle(
    url: str,
    track: SubtitleTrack,
    out_dir: Path,
    *,
    source_type: str = SOURCE_YOUTUBE,
    cookies: Optional[Path] = None,
    cancel_check: Optional[CancelCheck] = None,
    on_retry: Optional[RetryCallback] = None,
    ffmpeg_dir: Optional[str] = None,
) -> Path:
    """Baixa SOMENTE a faixa de legenda escolhida, em WebVTT. Não baixa áudio/vídeo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("subtitle*"):
        try:
            old.unlink()
        except OSError:
            pass

    def _once() -> Path:
        args = _base_args(cookies, ffmpeg_dir) + [
            "--skip-download",
            "--write-subs" if track.kind == "manual" else "--write-auto-subs",
            "--sub-langs", f"^{re.escape(track.lang)}$",
            "--sub-format", "vtt/best",
            "-o", str(out_dir / "subtitle.%(ext)s"),
        ]
        if "vtt" not in [e.lower() for e in track.exts]:
            args += ["--convert-subs", "vtt"]
        args.append(url)
        result = run_process(args, cancel_check=cancel_check, timeout=600)
        if result.returncode != 0:
            raise classify_ytdlp_error(result.stderr_text, source_type=source_type)
        files = sorted(out_dir.glob("subtitle*.vtt"))
        if not files:
            raise SubtitleError(
                "A legenda foi listada, mas não pôde ser baixada.", technical=result.stderr_text[-1500:]
            )
        return files[0]

    return retry_call(_once, cancel_check=cancel_check, on_retry=on_retry)


# =============================================================================
# Áudio
# =============================================================================

_PROG_RE = re.compile(r"\[PROG\]\s*([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)")
_FILE_RE = re.compile(r"\[FILE\]\s*(.+)$")


def _to_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    if not value or value.upper() in ("NA", "N/A", "NONE"):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_progress_line(line: str) -> Optional[DownloadProgress]:
    """Interpreta uma linha produzida pelo ``--progress-template`` deste módulo."""
    m = _PROG_RE.search(line)
    if not m:
        return None
    downloaded = _to_int(m.group(1)) or 0
    total = _to_int(m.group(2)) or _to_int(m.group(3))
    return DownloadProgress(
        downloaded_bytes=downloaded,
        total_bytes=total,
        fragment_index=_to_int(m.group(4)),
        fragment_count=_to_int(m.group(5)),
    )


def _find_downloaded(out_dir: Path, stem: str) -> Optional[Path]:
    cands = [
        p for p in out_dir.glob(f"{stem}.*")
        if p.is_file() and not p.name.endswith((".part", ".ytdl", ".tmp")) and ".part-" not in p.name
        and p.stat().st_size > 0
    ]
    return max(cands, key=lambda p: p.stat().st_size) if cands else None


def download_audio(
    url: str,
    out_dir: Path,
    *,
    source_type: str = SOURCE_YOUTUBE,
    cookies: Optional[Path] = None,
    expected_bytes: Optional[int] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
    on_process_start: Optional[Callable[[int], None]] = None,
    on_process_exit: Optional[Callable[[int], None]] = None,
    on_retry: Optional[RetryCallback] = None,
    ffmpeg_dir: Optional[str] = None,
    stem: str = "source_audio",
) -> Path:
    """Baixa somente o áudio para ``out_dir/<stem>.<ext>`` e retorna o caminho.

    * Se o arquivo já existir completo (retomada), é reutilizado.
    * Downloads interrompidos continuam do ponto em que pararam (``.part``).
    * ``progress_cb`` recebe o progresso REAL informado pelo yt-dlp.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _find_downloaded(out_dir, stem)
    if existing is not None:
        logger.info("Áudio já baixado anteriormente, reutilizando: %s", existing)
        return existing

    if expected_bytes:
        ensure_disk_space(out_dir, int(expected_bytes) + 200 * 1024 ** 2, purpose="baixar o áudio")

    final_path: dict[str, str] = {}

    def _on_line(line: str) -> None:
        m = _FILE_RE.search(line)
        if m:
            final_path["path"] = m.group(1).strip()
            return
        prog = parse_progress_line(line)
        if prog and progress_cb:
            progress_cb(prog)

    def _once() -> Path:
        args = _base_args(cookies, ffmpeg_dir) + [
            "-f", AUDIO_FORMAT_SELECTOR,
            "--concurrent-fragments", "4",
            "--continue",
            "--no-mtime",
            "--newline",
            "--progress",
            "--progress-template",
            "download:[PROG] %(progress.downloaded_bytes)s|%(progress.total_bytes)s|"
            "%(progress.total_bytes_estimate)s|%(progress.fragment_index)s|%(progress.fragment_count)s",
            "--print", "after_move:[FILE] %(filepath)s",
            "--no-simulate",
            "-o", str(out_dir / f"{stem}.%(ext)s"),
            url,
        ]
        result = run_process(
            args,
            on_stdout_line=_on_line,
            on_stderr_line=_on_line,
            cancel_check=cancel_check,
            on_start=on_process_start,
            on_exit=on_process_exit,
            idle_timeout=600,  # 10 min sem nenhuma saída = travado
        )
        if result.returncode != 0:
            raise classify_ytdlp_error(result.stderr_text, source_type=source_type)
        path = Path(final_path["path"]) if final_path.get("path") else None
        if path is None or not path.is_file():
            path = _find_downloaded(out_dir, stem)
        if path is None or path.stat().st_size == 0:
            raise DownloadError("O download terminou, mas o arquivo de áudio não foi encontrado.",
                                technical=result.stderr_text[-1500:])
        return path

    return retry_call(_once, attempts=4, cancel_check=cancel_check, on_retry=on_retry)
