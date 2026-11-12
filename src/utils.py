"""Utilitários compartilhados por todos os módulos.

Responsabilidades:

* Constantes do domínio (formatos, idiomas, modelos, estados de tarefa).
* Hierarquia de erros com mensagens compreensíveis para usuários leigos.
* Validação de caminhos locais e identificação de formatos.
* Formatação/conversão de timestamps.
* Gravação ATÔMICA de JSON (arquivo temporário + ``os.replace``) e leitura
  tolerante a falhas (com cópia de segurança ``.bak``).
* Lock entre processos baseado em arquivo (interface x worker).
* Execução de processos externos (FFmpeg, yt-dlp) com cancelamento,
  leitura linha a linha sem bloqueio e encerramento da árvore de processos.
* Identificação robusta de processos (PID + horário de criação), pois o PID
  sozinho pode ser reutilizado pelo sistema operacional.
* Novas tentativas com intervalos progressivos para falhas temporárias.

Este módulo não importa Streamlit nem faster-whisper.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar

import psutil

logger = logging.getLogger(__name__)

T = TypeVar("T")

IS_WINDOWS = os.name == "nt"

# =============================================================================
# Constantes do domínio
# =============================================================================

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm"})
AUDIO_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav", ".m4a", ".flac"})
SUPPORTED_EXTENSIONS: frozenset[str] = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

#: Formato interno de trabalho: WAV PCM 16 bits, mono, 16 kHz.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # bytes (pcm_s16le)
WAV_BYTES_PER_SECOND = TARGET_SAMPLE_RATE * TARGET_CHANNELS * TARGET_SAMPLE_WIDTH

#: Idiomas oferecidos na interface (códigos do Whisper). O Whisper não
#: distingue pt-BR de pt-PT: ambos usam "pt". O padrão do app é "pt".
LANGUAGES: dict[str, str] = {
    "auto": "Detectar automaticamente",
    "pt": "Português (Brasil / Portugal)",
    "en": "Inglês",
    "es": "Espanhol",
    "fr": "Francês",
    "it": "Italiano",
    "de": "Alemão",
    "ja": "Japonês",
    "ko": "Coreano",
    "zh": "Chinês",
    "ru": "Russo",
    "ar": "Árabe",
    "hi": "Hindi",
    "nl": "Holandês",
    "pl": "Polonês",
    "tr": "Turco",
    "uk": "Ucraniano",
}
DEFAULT_LANGUAGE = "pt"

#: Nomes de idioma por extenso (para o cabeçalho do TXT).
LANGUAGE_NAMES: dict[str, str] = {k: v for k, v in LANGUAGES.items() if k != "auto"}
LANGUAGE_NAMES["pt"] = "Português"


@dataclass(frozen=True)
class ModelInfo:
    """Descrição de um modelo Whisper oferecido na interface."""

    name: str
    label: str
    description: str
    #: Memória aproximada exigida (GB) por tipo de computação.
    ram_gb_int8: float
    vram_gb_float16: float
    vram_gb_int8_float16: float
    relative_speed: str


MODELS: dict[str, ModelInfo] = {
    "base": ModelInfo(
        "base", "base (rápido)",
        "Muito rápido e leve. Precisão modesta; bom para testes ou áudio limpo.",
        ram_gb_int8=0.6, vram_gb_float16=0.6, vram_gb_int8_float16=0.4,
        relative_speed="muito rápido",
    ),
    "small": ModelInfo(
        "small", "small (equilibrado — padrão)",
        "Bom equilíbrio entre precisão e velocidade. Recomendado para a maioria dos computadores.",
        ram_gb_int8=1.2, vram_gb_float16=1.2, vram_gb_int8_float16=0.8,
        relative_speed="rápido",
    ),
    "medium": ModelInfo(
        "medium", "medium (preciso)",
        "Mais preciso, porém bem mais lento na CPU. Ideal com GPU NVIDIA.",
        ram_gb_int8=2.6, vram_gb_float16=2.8, vram_gb_int8_float16=1.8,
        relative_speed="lento na CPU",
    ),
    "large-v3": ModelInfo(
        "large-v3", "large-v3 (máxima precisão)",
        "O mais preciso. Exige muitos recursos; na CPU pode levar várias vezes a duração do vídeo.",
        ram_gb_int8=4.8, vram_gb_float16=5.0, vram_gb_int8_float16=3.2,
        relative_speed="muito lento na CPU",
    ),
}
DEFAULT_MODEL = "small"

#: Frequência das marcações de tempo no TXT.
TIMESTAMP_MODES: dict[str, str] = {
    "segment": "A cada frase/segmento de fala (mais detalhado)",
    "30s": "A cada 30 segundos",
    "1min": "A cada 1 minuto",
    "5min": "A cada 5 minutos",
    "10min": "A cada 10 minutos",
    "15min": "A cada 15 minutos",
}
DEFAULT_TIMESTAMP_MODE = "segment"

DEVICE_CHOICES: dict[str, str] = {
    "auto": "Automático",
    "cpu": "CPU",
    "cuda": "GPU / CUDA (NVIDIA)",
}


class TaskState(str, Enum):
    """Estados possíveis de uma tarefa (gravados no checkpoint)."""

    PENDING = "pending"
    VALIDATING = "validating"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    SPLITTING = "splitting"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def label(self) -> str:
        return TASK_STATE_LABELS[self]

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATES


TASK_STATE_LABELS: dict[TaskState, str] = {
    TaskState.PENDING: "Aguardando início",
    TaskState.VALIDATING: "Verificando arquivo ou link",
    TaskState.DOWNLOADING: "Baixando áudio",
    TaskState.EXTRACTING_AUDIO: "Convertendo áudio",
    TaskState.SPLITTING: "Detectando pausas e dividindo em blocos",
    TaskState.LOADING_MODEL: "Carregando modelo de transcrição",
    TaskState.TRANSCRIBING: "Transcrevendo",
    TaskState.FINALIZING: "Organizando o texto e finalizando",
    TaskState.COMPLETED: "Concluída",
    TaskState.CANCEL_REQUESTED: "Cancelamento solicitado",
    TaskState.CANCELLED: "Cancelada",
    TaskState.FAILED: "Falhou",
}

TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {TaskState.COMPLETED, TaskState.CANCELLED, TaskState.FAILED}
)
#: Estados a partir dos quais uma tarefa pode ser retomada.
RESUMABLE_STATES: frozenset[TaskState] = frozenset(
    {TaskState.CANCELLED, TaskState.FAILED}
) | (frozenset(TaskState) - TERMINAL_STATES)


# =============================================================================
# Erros com mensagens amigáveis
# =============================================================================


class TranscriberError(Exception):
    """Erro base do aplicativo.

    ``user_message`` é exibida na interface (linguagem simples);
    ``technical`` vai para o log.
    """

    code = "error"
    default_message = "Ocorreu um erro inesperado durante o processamento."

    def __init__(self, user_message: str | None = None, *, technical: str | None = None) -> None:
        self.user_message = user_message or self.default_message
        self.technical = technical
        super().__init__(self.user_message if not technical else f"{self.user_message} | {technical}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "type": type(self).__name__,
            "message": self.user_message,
            "technical": self.technical,
        }


class InvalidInputError(TranscriberError):
    code = "invalid_input"
    default_message = "O arquivo ou link informado não é válido."


class UnsupportedFormatError(InvalidInputError):
    code = "unsupported_format"
    default_message = "Formato de arquivo não suportado."


class FFmpegNotFoundError(TranscriberError):
    code = "ffmpeg_not_found"
    default_message = (
        "O FFmpeg não foi encontrado. Instale o FFmpeg e adicione a pasta 'bin' dele ao PATH "
        "do Windows (veja o README), depois feche e abra o aplicativo novamente."
    )


class AudioProcessingError(TranscriberError):
    code = "audio_processing"
    default_message = "Não foi possível processar o áudio deste arquivo."


class CorruptedMediaError(AudioProcessingError):
    code = "corrupted_media"
    default_message = "O arquivo parece estar corrompido ou não contém uma faixa de áudio legível."


class DiskSpaceError(TranscriberError):
    code = "disk_space"
    default_message = "Não há espaço livre suficiente no disco para processar este arquivo."


class InsufficientMemoryError(TranscriberError):
    code = "insufficient_memory"
    default_message = (
        "Memória RAM insuficiente para este modelo. Feche outros programas ou escolha um modelo menor."
    )


class NetworkError(TranscriberError):
    """Falha temporária de conexão (elegível para novas tentativas)."""

    code = "network"
    default_message = "Falha de conexão com a internet. Verifique sua rede e tente novamente."


class DownloadError(TranscriberError):
    code = "download"
    default_message = "Não foi possível baixar o conteúdo."


class ContentUnavailableError(DownloadError):
    code = "content_unavailable"
    default_message = "O vídeo não está disponível (removido, privado, expirado ou inexistente)."


class AuthenticationRequiredError(DownloadError):
    code = "auth_required"
    default_message = (
        "Este conteúdo exige login. Informe um arquivo de cookies (cookies.txt) de uma sessão "
        "sua autorizada e tente novamente."
    )


class ProtectedContentError(DownloadError):
    code = "protected_content"
    default_message = (
        "Este conteúdo é protegido (DRM, assinatura paga ou acesso restrito) e não pode ser "
        "baixado por este aplicativo."
    )


class SubtitleError(TranscriberError):
    code = "subtitle"
    default_message = "Não foi possível obter ou ler as legendas do vídeo."


class GPUError(TranscriberError):
    code = "gpu"
    default_message = "A GPU não pôde ser utilizada. O processamento pode ser feito pela CPU."


class ModelLoadError(TranscriberError):
    code = "model_load"
    default_message = (
        "Não foi possível carregar o modelo de transcrição. Verifique a conexão (no primeiro uso o "
        "modelo é baixado) e o espaço em disco."
    )


class CheckpointError(TranscriberError):
    code = "checkpoint"
    default_message = "Não foi possível gravar o progresso da tarefa no disco."


class CorruptedJSONError(CheckpointError):
    code = "corrupted_json"
    default_message = "O arquivo de progresso da tarefa está corrompido e não pôde ser recuperado."


class WorkerError(TranscriberError):
    code = "worker"
    default_message = "O processo de transcrição em segundo plano falhou."


class TaskCancelledError(TranscriberError):
    code = "cancelled"
    default_message = "A tarefa foi cancelada pelo usuário."


class ProcessTimeoutError(TranscriberError):
    code = "timeout"
    default_message = "Uma etapa demorou mais do que o esperado e foi interrompida."


def friendly_error(exc: BaseException) -> str:
    """Converte qualquer exceção em uma mensagem compreensível."""
    if isinstance(exc, TranscriberError):
        return exc.user_message
    if isinstance(exc, MemoryError):
        return InsufficientMemoryError.default_message
    if isinstance(exc, PermissionError):
        return (
            "O Windows negou acesso a um arquivo ou pasta. Verifique se o arquivo não está aberto em "
            "outro programa e se você tem permissão de leitura/escrita."
        )
    if isinstance(exc, FileNotFoundError):
        return "Um arquivo necessário não foi encontrado. Ele pode ter sido movido ou excluído."
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return DiskSpaceError.default_message
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 112:  # ERROR_DISK_FULL
        return DiskSpaceError.default_message
    return f"Ocorreu um erro inesperado: {type(exc).__name__}. Detalhes técnicos foram salvos no log."


def error_to_dict(exc: BaseException) -> dict[str, Any]:
    """Representação serializável de um erro, para o checkpoint."""
    if isinstance(exc, TranscriberError):
        data = exc.to_dict()
    else:
        data = {
            "code": "unexpected",
            "type": type(exc).__name__,
            "message": friendly_error(exc),
            "technical": repr(exc),
        }
    data["time"] = now_iso()
    return data


# =============================================================================
# Logging
# =============================================================================

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(process)d %(name)s: %(message)s"


def setup_logging(log_file: Path | None = None, level: int = logging.INFO, *, console: bool = True) -> None:
    """Configura o logging raiz (arquivo rotativo UTF-8 e/ou console).

    Pode ser chamada várias vezes (ex.: reruns do Streamlit) sem duplicar handlers.
    """
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        target = str(log_file.resolve())
        already = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == target
            for h in root.handlers
        )
        if not already:
            fh = RotatingFileHandler(target, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
            fh.setFormatter(formatter)
            root.addHandler(fh)

    if console and not any(getattr(h, "_transcritor_console", False) for h in root.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch._transcritor_console = True  # type: ignore[attr-defined]
        root.addHandler(ch)


# =============================================================================
# Datas, IDs e timestamps
# =============================================================================


def now_iso() -> str:
    """Data/hora local atual em ISO 8601 (com fuso)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def generate_task_id() -> str:
    """Identificador único e ordenável de tarefa: ``AAAAMMDD-HHMMSS-xxxxxxxx``."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"


_TASK_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")


def is_valid_task_id(task_id: str) -> bool:
    """Impede caminhos maliciosos/inesperados ao usar o ID como nome de pasta."""
    return bool(_TASK_ID_RE.match(task_id or ""))


def format_timestamp(seconds: float, style: str = "txt") -> str:
    """Formata segundos como timestamp.

    ``style``:
      * ``"txt"`` → ``HH:MM:SS``
      * ``"srt"`` → ``HH:MM:SS,mmm``
      * ``"vtt"`` → ``HH:MM:SS.mmm``
    """
    if seconds is None or seconds != seconds:  # None ou NaN
        seconds = 0.0
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    if style == "txt":
        total_ms = (total_ms // 1000) * 1000  # trunca para o segundo (nunca adianta o tempo)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    if style == "txt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    sep = "," if style == "srt" else "."
    if style not in ("srt", "vtt"):
        raise ValueError(f"Estilo de timestamp desconhecido: {style}")
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


_TS_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[.,](\d{1,3}))?\s*$")


def parse_timestamp(value: str) -> float:
    """Converte ``HH:MM:SS.mmm``, ``HH:MM:SS,mmm`` ou ``MM:SS.mmm`` em segundos."""
    match = _TS_RE.match(value or "")
    if not match:
        raise ValueError(f"Timestamp inválido: {value!r}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    secs = int(match.group(3))
    frac = match.group(4) or "0"
    ms = int(frac.ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + secs + ms / 1000.0


def format_duration(seconds: float) -> str:
    """Duração legível: ``2h 05min 03s``, ``12min 40s``, ``8s``."""
    seconds = max(0, int(round(seconds or 0)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}min {s:02d}s"
    if m:
        return f"{m}min {s:02d}s"
    return f"{s}s"


def format_bytes(num: float) -> str:
    """Tamanho legível em unidades binárias (KB, MB, GB...)."""
    num = float(num or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


# =============================================================================
# Caminhos e validação de arquivos
# =============================================================================


def clean_user_path(raw: str) -> str:
    """Remove aspas e espaços que o Windows adiciona em "Copiar como caminho"."""
    value = (raw or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value


def classify_media_extension(path: Path | str) -> Optional[str]:
    """Retorna ``"video"``, ``"audio"`` ou ``None`` com base na extensão."""
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return None


def validate_local_file(raw_path: str | Path) -> Path:
    """Valida um caminho local informado pelo usuário e retorna o ``Path`` absoluto.

    Verifica: caminho preenchido, existência, se é arquivo, extensão suportada,
    tamanho maior que zero e permissão de leitura. O arquivo é apenas aberto
    para leitura de 1 byte — nunca é modificado.
    """
    text = clean_user_path(str(raw_path) if raw_path is not None else "")
    if not text:
        raise InvalidInputError("Informe o caminho completo do arquivo (ex.: C:\\Videos\\gravacao.mp4).")

    try:
        path = Path(os.path.expandvars(os.path.expanduser(text)))
        if not path.is_absolute():
            path = path.resolve()
    except (OSError, ValueError) as exc:
        raise InvalidInputError("O caminho informado é inválido.", technical=repr(exc)) from exc

    try:
        exists = path.exists()
    except OSError as exc:  # ex.: nomes inválidos no Windows
        raise InvalidInputError("O caminho informado é inválido.", technical=repr(exc)) from exc

    if not exists:
        raise InvalidInputError(f"Arquivo não encontrado: {path}")
    if path.is_dir():
        raise InvalidInputError("O caminho informado é uma pasta. Selecione um arquivo de vídeo ou áudio.")
    if not path.is_file():
        raise InvalidInputError(f"O caminho não aponta para um arquivo comum: {path}")

    if classify_media_extension(path) is None:
        allowed = ", ".join(sorted(e.lstrip(".").upper() for e in SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"O formato '{path.suffix or '(sem extensão)'}' não é suportado. Formatos aceitos: {allowed}."
        )

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InvalidInputError("Não foi possível ler as informações do arquivo.", technical=repr(exc)) from exc
    if size == 0:
        raise CorruptedMediaError("O arquivo está vazio (0 bytes).")

    try:
        with open(path, "rb") as handle:
            handle.read(1)
    except PermissionError as exc:
        raise InvalidInputError(
            "Sem permissão para ler o arquivo. Ele pode estar aberto em outro programa ou protegido.",
            technical=repr(exc),
        ) from exc
    except OSError as exc:
        raise InvalidInputError("O arquivo não pôde ser lido.", technical=repr(exc)) from exc

    return path


_WIN_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(name: str, max_length: int = 80, fallback: str = "transcricao") -> str:
    """Gera um nome de arquivo/pasta seguro para o Windows."""
    value = _INVALID_CHARS_RE.sub("_", name or "")
    value = re.sub(r"\s+", " ", value).strip().strip(".")
    value = re.sub(r"_+", "_", value)
    if len(value) > max_length:
        value = value[:max_length].rstrip(" ._")
    if not value:
        value = fallback
    if value.split(".")[0].upper() in _WIN_RESERVED:
        value = f"_{value}"
    return value


def unique_path(base: Path) -> Path:
    """Retorna ``base`` ou ``base (2)``, ``base (3)``... sem sobrescrever nada existente."""
    if not base.exists():
        return base
    stem, suffix = (base.stem, base.suffix) if base.suffix else (base.name, "")
    for i in range(2, 10_000):
        candidate = base.with_name(f"{stem} ({i}){suffix}")
        if not candidate.exists():
            return candidate
    raise CheckpointError(f"Não foi possível criar um nome único para {base}")


def unique_output_dir(base_dir: Path, name: str) -> Path:
    """Pasta de resultados ``base_dir/nome`` única (não sobrescreve resultados anteriores)."""
    return unique_path(Path(base_dir) / sanitize_filename(name))


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def open_folder(path: Path | str) -> None:
    """Abre uma pasta no explorador de arquivos do sistema."""
    path = Path(path)
    if not path.exists():
        raise InvalidInputError(f"A pasta não existe: {path}")
    if IS_WINDOWS:
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


# =============================================================================
# Disco e memória
# =============================================================================


def _existing_ancestor(path: Path) -> Path:
    path = Path(path).resolve()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def get_free_disk_space(path: Path | str) -> int:
    """Bytes livres no volume que contém ``path`` (mesmo que ``path`` ainda não exista)."""
    return shutil.disk_usage(_existing_ancestor(Path(path))).free


def ensure_disk_space(path: Path | str, required_bytes: int, *, purpose: str = "processar o arquivo") -> None:
    """Lança ``DiskSpaceError`` se não houver ``required_bytes`` livres."""
    free = get_free_disk_space(path)
    if free < required_bytes:
        raise DiskSpaceError(
            f"Espaço insuficiente para {purpose}: são necessários cerca de {format_bytes(required_bytes)}, "
            f"mas há apenas {format_bytes(free)} livres em {_existing_ancestor(Path(path)).anchor or path}.",
            technical=f"required={required_bytes} free={free}",
        )


def estimate_wav_bytes(duration_seconds: float) -> int:
    """Tamanho do WAV de trabalho (16 kHz, mono, 16 bits) para uma duração."""
    return int(max(0.0, duration_seconds) * WAV_BYTES_PER_SECOND) + 4096


def get_available_ram() -> int:
    """Bytes de RAM disponíveis no momento."""
    return int(psutil.virtual_memory().available)


# =============================================================================
# JSON atômico
# =============================================================================


def _replace_with_retry(src: str, dst: str, attempts: int = 40, delay: float = 0.05) -> None:
    """``os.replace`` com novas tentativas.

    No Windows, ``os.replace`` falha com ``PermissionError`` se outro processo
    estiver lendo o destino naquele instante; o leitor solta o arquivo em
    milissegundos, então algumas tentativas resolvem.
    """
    last_exc: Optional[BaseException] = None
    for _ in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(delay)
    raise CheckpointError(technical=f"os.replace falhou após {attempts} tentativas: {last_exc!r}")


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """Grava texto de forma atômica: arquivo temporário → fsync → ``os.replace``.

    Se o processo for interrompido no meio, o arquivo anterior permanece íntegro.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        with open(tmp, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(str(tmp), str(path))
    except CheckpointError:
        raise
    except OSError as exc:
        raise CheckpointError(
            DiskSpaceError.default_message if getattr(exc, "errno", None) == errno.ENOSPC else None,
            technical=f"Falha ao gravar {path}: {exc!r}",
        ) from exc
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def backup_path(path: Path | str) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".bak")


def atomic_write_json(path: Path | str, data: Any, *, keep_backup: bool = True) -> None:
    """Grava JSON de forma atômica (UTF-8, legível).

    Com ``keep_backup=True``, também mantém ``<arquivo>.bak`` com o mesmo
    conteúdo válido, usado se o principal for corrompido por falha de disco.
    """
    text = json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)
    atomic_write_text(path, text)
    if keep_backup:
        try:
            atomic_write_text(backup_path(path), text)
        except CheckpointError:  # o principal já foi gravado; o backup é secundário
            logger.warning("Não foi possível atualizar a cópia de segurança de %s", path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Objeto não serializável em JSON: {type(obj).__name__}")


_MISSING = object()


def read_json(path: Path | str, default: Any = _MISSING, *, use_backup: bool = True, attempts: int = 10) -> Any:
    """Lê JSON tolerando escrita concorrente e corrupção.

    1. Tenta o arquivo principal (com novas tentativas para ``PermissionError``).
    2. Se estiver corrompido/ausente, tenta ``<arquivo>.bak``.
    3. Se nada for válido: retorna ``default`` (quando informado) ou lança
       ``CorruptedJSONError`` / ``FileNotFoundError``.
    """
    path = Path(path)
    candidates = [path] + ([backup_path(path)] if use_backup else [])
    corrupted = False
    for candidate in candidates:
        for attempt in range(attempts):
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except FileNotFoundError:
                break
            except PermissionError:
                time.sleep(0.05)
                continue
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                corrupted = True
                logger.warning("JSON corrompido em %s: %s", candidate, exc)
                break
    if default is not _MISSING:
        return default
    if corrupted:
        raise CorruptedJSONError(technical=f"Nenhuma cópia válida de {path}")
    raise FileNotFoundError(str(path))


# =============================================================================
# Identificação de processos
# =============================================================================


def get_process_create_time(pid: int) -> Optional[float]:
    """Horário de criação do processo (epoch) ou ``None`` se não existir."""
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def current_process_identity() -> dict[str, Any]:
    """Identidade do processo atual, gravada em checkpoints e locks."""
    pid = os.getpid()
    return {"pid": pid, "create_time": get_process_create_time(pid), "host": _hostname()}


def _hostname() -> str:
    try:
        import socket

        return socket.gethostname()
    except OSError:
        return ""


def is_process_alive(
    pid: Optional[int],
    create_time: Optional[float] = None,
    *,
    cmdline_marker: Optional[str] = None,
    tolerance: float = 1.0,
) -> bool:
    """Verifica se o processo ``pid`` existe E é o mesmo processo registrado.

    O PID pode ser reutilizado pelo Windows; por isso compara também o horário
    de criação (``create_time``) e, opcionalmente, procura ``cmdline_marker``
    (ex.: o ID da tarefa) na linha de comando.
    """
    if not pid:
        return False
    try:
        proc = psutil.Process(int(pid))
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if create_time is not None and abs(proc.create_time() - float(create_time)) > tolerance:
            return False
        if cmdline_marker:
            try:
                cmdline = " ".join(proc.cmdline())
            except (psutil.AccessDenied, psutil.ZombieProcess):
                cmdline = ""
            if cmdline and cmdline_marker not in cmdline:
                return False
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, OSError):
        return False


def kill_process_tree(pid: int, *, timeout: float = 5.0, include_parent: bool = True) -> list[int]:
    """Encerra um processo e todos os seus descendentes (terminate → kill).

    Retorna a lista de PIDs encerrados. Nunca lança exceção por processo já finalizado.
    """
    killed: list[int] = []
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, ValueError):
        return killed
    try:
        procs = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        procs = []
    if include_parent:
        procs.append(parent)
    for proc in procs:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(alive, timeout=timeout)
    killed = [p.pid for p in procs]
    return killed


# =============================================================================
# Lock entre processos (baseado em arquivo)
# =============================================================================


class FileLock:
    """Lock exclusivo entre processos usando criação atômica de arquivo (``O_EXCL``).

    Usado para garantir que apenas um processo escreva em um recurso
    compartilhado por vez (ex.: o worker atualizando o checkpoint enquanto a
    interface grava um pedido de retomada). Locks "órfãos" (de processos que
    morreram) são detectados pelo PID + horário de criação e removidos.

    Exemplo::

        with FileLock(task_dir / "checkpoint.lock"):
            ...
    """

    def __init__(self, path: Path | str, *, timeout: float = 10.0, poll: float = 0.05, stale_after: float = 120.0):
        self.path = Path(path)
        self.timeout = timeout
        self.poll = poll
        self.stale_after = stale_after
        self._held = False

    def _is_stale(self) -> bool:
        try:
            # Arquivo vazio = o dono acabou de criá-lo (O_EXCL) e ainda não gravou o conteúdo:
            # NÃO é lock órfão. O JSONDecodeError abaixo cai no teste por idade do arquivo.
            info = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(info, dict):
                raise json.JSONDecodeError("não é um objeto", "", 0)
        except (OSError, json.JSONDecodeError):
            # Arquivo sendo criado ou ilegível: considera órfão apenas se for antigo.
            try:
                return time.time() - self.path.stat().st_mtime > self.stale_after
            except OSError:
                return False
        if not is_process_alive(info.get("pid"), info.get("create_time")):
            return True
        return time.time() - float(info.get("acquired_at", time.time())) > self.stale_after

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({**current_process_identity(), "acquired_at": time.time()})
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                self._held = True
                return
            except FileExistsError:
                if self._is_stale():
                    logger.warning("Removendo lock órfão: %s", self.path)
                    try:
                        self.path.unlink()
                    except OSError:
                        pass
                    continue
            except PermissionError:
                pass  # Windows: arquivo em remoção por outro processo
            if time.monotonic() >= deadline:
                raise CheckpointError(
                    "Outro processo está usando os arquivos desta tarefa. Aguarde alguns segundos.",
                    technical=f"Timeout ao obter lock {self.path}",
                )
            time.sleep(self.poll)

    def release(self) -> None:
        if self._held:
            self._held = False
            for _ in range(20):
                try:
                    self.path.unlink()
                    return
                except FileNotFoundError:
                    return
                except PermissionError:
                    time.sleep(0.05)

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# =============================================================================
# Execução de processos externos (FFmpeg, yt-dlp)
# =============================================================================

#: Não abre janelas de console para processos filhos no Windows.
CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0


def subprocess_env(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Ambiente para processos filhos com saída UTF-8 garantida."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if extra:
        env.update(extra)
    return env


@dataclass
class ProcessResult:
    returncode: int
    stdout_tail: list[str] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)
    duration: float = 0.0

    @property
    def stderr_text(self) -> str:
        return "\n".join(self.stderr_tail)

    @property
    def stdout_text(self) -> str:
        return "\n".join(self.stdout_tail)


def _pump(stream: Any, name: str, out: "queue.Queue[tuple[str, Optional[str]]]") -> None:
    try:
        for line in iter(stream.readline, ""):
            out.put((name, line.rstrip("\r\n")))
    except (ValueError, OSError):
        pass
    finally:
        out.put((name, None))


def run_process(
    args: Sequence[str],
    *,
    on_stdout_line: Optional[Callable[[str], None]] = None,
    on_stderr_line: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    on_start: Optional[Callable[[int], None]] = None,
    on_exit: Optional[Callable[[int], None]] = None,
    timeout: Optional[float] = None,
    idle_timeout: Optional[float] = None,
    cwd: Optional[Path | str] = None,
    env: Optional[dict[str, str]] = None,
    tail_lines: int = 200,
    poll_interval: float = 0.2,
) -> ProcessResult:
    """Executa um processo externo sem ``shell=True`` e sem bloquear o cancelamento.

    * stdout e stderr são lidos linha a linha em threads (evita deadlock de pipe
      cheio); os callbacks são chamados na thread atual.
    * ``cancel_check()`` é consultado a cada ``poll_interval``; se retornar
      ``True``, a árvore de processos é encerrada e ``TaskCancelledError`` é lançado.
    * ``timeout`` limita a duração total; ``idle_timeout`` limita o tempo sem
      nenhuma saída (processos travados). Ambos lançam ``ProcessTimeoutError``.
    * ``on_start(pid)`` / ``on_exit(pid)`` permitem registrar o PID filho no
      checkpoint para limpeza de processos órfãos.
    * Nunca deixa o processo vivo ao sair (inclusive em exceções).
    """
    start = time.monotonic()
    stdout_tail: deque[str] = deque(maxlen=tail_lines)
    stderr_tail: deque[str] = deque(maxlen=tail_lines)
    lines: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()

    try:
        proc = subprocess.Popen(
            list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(cwd) if cwd else None,
            env=env or subprocess_env(),
            creationflags=CREATE_NO_WINDOW,
        )
    except FileNotFoundError as exc:
        raise TranscriberError(
            f"Programa externo não encontrado: {Path(args[0]).name}", technical=repr(exc)
        ) from exc
    except OSError as exc:
        raise TranscriberError(
            f"Não foi possível iniciar o programa externo {Path(args[0]).name}.", technical=repr(exc)
        ) from exc

    if on_start:
        try:
            on_start(proc.pid)
        except Exception:  # noqa: BLE001 - registro de PID não pode derrubar a execução
            logger.exception("Falha no callback on_start")

    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, "out", lines), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, "err", lines), daemon=True),
    ]
    for t in threads:
        t.start()

    open_streams = 2
    last_output = time.monotonic()
    last_cancel_check = 0.0
    try:
        while open_streams > 0 or proc.poll() is None:
            try:
                name, line = lines.get(timeout=poll_interval)
            except queue.Empty:
                name, line = "", None
            else:
                if line is None:
                    open_streams -= 1
                else:
                    last_output = time.monotonic()
                    if name == "out":
                        stdout_tail.append(line)
                        if on_stdout_line:
                            on_stdout_line(line)
                    else:
                        stderr_tail.append(line)
                        if on_stderr_line:
                            on_stderr_line(line)

            now = time.monotonic()
            # Consulta o cancelamento no máximo a cada poll_interval (a saída
            # pode ter centenas de linhas por segundo).
            if cancel_check and now - last_cancel_check >= poll_interval:
                last_cancel_check = now
                if cancel_check():
                    raise TaskCancelledError()
            if timeout is not None and now - start > timeout:
                raise ProcessTimeoutError(technical=f"timeout={timeout}s args={args[:3]}")
            if idle_timeout is not None and now - last_output > idle_timeout and proc.poll() is None:
                raise ProcessTimeoutError(
                    "Um programa externo parou de responder e foi encerrado.",
                    technical=f"idle_timeout={idle_timeout}s args={args[:3]}",
                )
            if open_streams == 0 and proc.poll() is None:
                # Streams fechados mas processo ainda finalizando.
                try:
                    proc.wait(timeout=poll_interval)
                except subprocess.TimeoutExpired:
                    pass
        returncode = proc.wait()
    except BaseException:
        kill_process_tree(proc.pid)
        raise
    finally:
        for t in threads:
            t.join(timeout=2)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        if on_exit:
            try:
                on_exit(proc.pid)
            except Exception:  # noqa: BLE001
                logger.exception("Falha no callback on_exit")

    return ProcessResult(
        returncode=returncode,
        stdout_tail=list(stdout_tail),
        stderr_tail=list(stderr_tail),
        duration=time.monotonic() - start,
    )


# =============================================================================
# Novas tentativas com intervalos progressivos
# =============================================================================


def interruptible_sleep(seconds: float, cancel_check: Optional[Callable[[], bool]] = None, step: float = 0.25) -> None:
    """``time.sleep`` que pode ser interrompido por cancelamento."""
    end = time.monotonic() + max(0.0, seconds)
    while True:
        if cancel_check and cancel_check():
            raise TaskCancelledError()
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(step, remaining))


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 3.0,
    max_delay: float = 60.0,
    retry_on: tuple[type[BaseException], ...] = (NetworkError,),
    cancel_check: Optional[Callable[[], bool]] = None,
    on_retry: Optional[Callable[[int, float, BaseException], None]] = None,
) -> T:
    """Executa ``func`` com novas tentativas para erros temporários.

    Intervalos progressivos (exponenciais): 3 s, 6 s, 12 s... limitados a ``max_delay``.
    ``on_retry(tentativa, espera, erro)`` permite informar a interface.
    """
    last: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except retry_on as exc:  # type: ignore[misc]
            last = exc
            if attempt >= attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.warning("Tentativa %d/%d falhou (%s). Nova tentativa em %.0fs.", attempt, attempts, exc, delay)
            if on_retry:
                on_retry(attempt, delay, exc)
            interruptible_sleep(delay, cancel_check)
    assert last is not None
    raise last


def chunked(items: Iterable[T], size: int) -> Iterable[list[T]]:
    """Agrupa um iterável em listas de até ``size`` itens."""
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
