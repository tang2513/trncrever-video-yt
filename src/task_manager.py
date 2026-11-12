"""Gerenciamento de tarefas: arquivos em disco, checkpoints, processos e cancelamento.

Este módulo é usado tanto pela interface (``app.py``) quanto pelo worker
(``src/worker.py``). Ele não importa Streamlit nem faster-whisper.

Layout de uma tarefa (``temp/tasks/<ID>/``)::

    config.json        configurações (escrito SÓ pela interface, antes de iniciar/retomar)
    checkpoint.json    estado e progresso (escrito pelo worker; exceções abaixo)
    manifest.json      plano de blocos + blocos concluídos (escrito SÓ pelo worker)
    cancel.json        pedido de cancelamento (escrito SÓ pela interface)
    checkpoint.lock    lock de escrita do checkpoint
    worker.lock        lock mantido pelo worker durante toda a execução
    launch.lock        lock curto para impedir inicializações duplicadas
    audio/ chunks/ segments/ work/   arquivos temporários e resultados parciais

Quem escreve o quê (propriedade das atualizações):

* **Interface**: ``config.json``; ``cancel.json``; o checkpoint INICIAL (antes
  de existir worker); o campo ``worker`` no instante do lançamento; a
  reinicialização do estado ao retomar (só com o worker inativo); e a marcação
  de "falhou/cancelado" quando o worker morreu sem concluir (``reconcile_task``).
* **Worker**: todo o resto do checkpoint e todo o ``manifest.json``.

Toda escrita no checkpoint passa por ``CheckpointStore``: lock de arquivo
(entre processos) + lock de thread + gravação atômica (``os.replace``). Assim a
interface e o worker nunca escrevem ao mesmo tempo, e o último estado válido
nunca se perde se o processo for interrompido durante a gravação.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional

from . import LOGS_DIR, PROJECT_ROOT, TASKS_DIR
from .audio_processor import DEFAULT_CHUNK_SECONDS, SilenceSettings
from .downloader import SOURCE_LOCAL, SOURCE_YOUTUBE
from .subtitles import DEFAULT_SUBTITLE_MODE, SUBTITLE_MODES
from .utils import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_TIMESTAMP_MODE,
    DEVICE_CHOICES,
    LANGUAGES,
    MODELS,
    RESUMABLE_STATES,
    TERMINAL_STATES,
    TIMESTAMP_MODES,
    CheckpointError,
    CorruptedJSONError,
    FileLock,
    InvalidInputError,
    TaskState,
    WorkerError,
    atomic_write_json,
    error_to_dict,
    generate_task_id,
    get_process_create_time,
    is_process_alive,
    is_valid_task_id,
    kill_process_tree,
    now_iso,
    read_json,
    subprocess_env,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Apenas um worker por vez: duas transcrições simultâneas disputariam CPU/RAM/GPU.
MAX_ACTIVE_WORKERS = 1

# =============================================================================
# Etapas exibidas na interface (as 9 etapas do projeto)
# =============================================================================

STEP_ORDER: tuple[str, ...] = (
    "validate", "subtitles", "download", "convert", "split", "load_model", "transcribe", "organize", "finish",
)
STEP_LABELS: dict[str, str] = {
    "validate": "1. Verificação do arquivo ou link",
    "subtitles": "2. Consulta de legendas",
    "download": "3. Download do áudio",
    "convert": "4. Conversão do áudio",
    "split": "5. Detecção de pausas e divisão em blocos",
    "load_model": "6. Carregamento do modelo",
    "transcribe": "7. Transcrição",
    "organize": "8. Organização do texto",
    "finish": "9. Finalização",
}
#: Peso de cada etapa no progresso geral. O progresso geral é a média ponderada
#: das frações REAIS de cada etapa (etapas puladas saem do cálculo).
STEP_WEIGHTS: dict[str, float] = {
    "validate": 0.01, "subtitles": 0.01, "download": 0.08, "convert": 0.04, "split": 0.03,
    "load_model": 0.02, "transcribe": 0.76, "organize": 0.04, "finish": 0.01,
}
STEP_PENDING, STEP_RUNNING, STEP_DONE, STEP_SKIPPED = "pending", "running", "done", "skipped"

STATE_TO_STEP: dict[TaskState, str] = {
    TaskState.VALIDATING: "validate",
    TaskState.DOWNLOADING: "download",
    TaskState.EXTRACTING_AUDIO: "convert",
    TaskState.SPLITTING: "split",
    TaskState.LOADING_MODEL: "load_model",
    TaskState.TRANSCRIBING: "transcribe",
    TaskState.FINALIZING: "organize",
}

# =============================================================================
# Configuração da tarefa
# =============================================================================


@dataclass
class TaskConfig:
    """Configurações imutáveis de uma tarefa (gravadas em ``config.json``)."""

    task_id: str
    source_type: str                 # "local" | "youtube" | "twitch"
    source: str                      # caminho local ou URL normalizada
    title: str
    model: str = DEFAULT_MODEL
    language: str = DEFAULT_LANGUAGE  # "auto" ou código do Whisper
    device: str = "auto"             # "auto" | "cpu" | "cuda"
    allow_cpu_fallback: bool = True
    timestamp_mode: str = DEFAULT_TIMESTAMP_MODE
    subtitle_mode: str = DEFAULT_SUBTITLE_MODE
    cookies_file: Optional[str] = None
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    silence: dict[str, float] = field(default_factory=lambda: SilenceSettings().to_dict())
    use_silence_split: bool = True
    beam_size: int = 5
    overwrite_existing: bool = False
    keep_temp: bool = False
    created_at: str = field(default_factory=now_iso)
    schema: int = SCHEMA_VERSION

    def validate(self) -> None:
        if not is_valid_task_id(self.task_id):
            raise InvalidInputError("Identificador de tarefa inválido.")
        if self.model not in MODELS:
            raise InvalidInputError(f"Modelo desconhecido: {self.model}")
        if self.language not in LANGUAGES:
            raise InvalidInputError(f"Idioma desconhecido: {self.language}")
        if self.device not in DEVICE_CHOICES:
            raise InvalidInputError(f"Dispositivo desconhecido: {self.device}")
        if self.timestamp_mode not in TIMESTAMP_MODES:
            raise InvalidInputError(f"Frequência de marcações inválida: {self.timestamp_mode}")
        if self.subtitle_mode not in SUBTITLE_MODES:
            raise InvalidInputError(f"Modo de legendas inválido: {self.subtitle_mode}")
        if self.source_type not in (SOURCE_LOCAL, SOURCE_YOUTUBE, "twitch"):
            raise InvalidInputError(f"Tipo de origem inválido: {self.source_type}")
        if not 60 <= float(self.chunk_seconds) <= 1800:
            raise InvalidInputError("O tamanho do bloco deve ficar entre 1 e 30 minutos.")
        if not 1 <= int(self.beam_size) <= 10:
            raise InvalidInputError("O beam size deve ficar entre 1 e 10.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class TaskPaths:
    task_id: str
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def cancel(self) -> Path:
        return self.root / "cancel.json"

    @property
    def checkpoint_lock(self) -> Path:
        return self.root / "checkpoint.lock"

    @property
    def worker_lock(self) -> Path:
        return self.root / "worker.lock"

    @property
    def launch_lock(self) -> Path:
        return self.root / "launch.lock"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def chunks_dir(self) -> Path:
        return self.root / "chunks"

    @property
    def segments_dir(self) -> Path:
        return self.root / "segments"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def log_file(self) -> Path:
        return LOGS_DIR / f"worker_{self.task_id}.log"

    def segments_file(self, index: int) -> Path:
        return self.segments_dir / f"segments_{index:04d}.json"

    @property
    def subtitles_segments_file(self) -> Path:
        return self.segments_dir / "subtitles.json"


def task_paths(task_id: str) -> TaskPaths:
    """Caminhos da tarefa. Rejeita IDs que possam escapar da pasta ``temp/tasks``."""
    if not is_valid_task_id(task_id):
        raise InvalidInputError("Identificador de tarefa inválido.")
    return TaskPaths(task_id, TASKS_DIR / task_id)


# =============================================================================
# Checkpoint
# =============================================================================


def initial_checkpoint(config: TaskConfig) -> dict[str, Any]:
    """Estado inicial gravado pela interface ao criar a tarefa."""
    steps = {s: {"status": STEP_PENDING, "fraction": 0.0} for s in STEP_ORDER}
    if config.source_type != SOURCE_YOUTUBE:
        steps["subtitles"]["status"] = STEP_SKIPPED
    if config.source_type == SOURCE_LOCAL:
        steps["download"]["status"] = STEP_SKIPPED
    paths = task_paths(config.task_id)
    return {
        "schema": SCHEMA_VERSION,
        "task_id": config.task_id,
        "title": config.title,
        "source_type": config.source_type,
        "source": config.source,
        "source_path": config.source if config.source_type == SOURCE_LOCAL else None,
        "model": config.model,
        "language_requested": config.language,
        "language_detected": None,
        "language_probability": None,
        "device_requested": config.device,
        "device": None,
        "state": TaskState.PENDING.value,
        "step": None,
        "message": "Tarefa criada. Aguardando o início do processamento.",
        "steps": steps,
        "overall_progress": 0.0,
        "mode": None,                      # "audio" | "subtitles" (definido pelo worker)
        "total_blocks": 0,
        "completed_blocks": [],
        "current_block": None,
        "blocks": [],
        "audio_duration": None,
        "transcribed_seconds": 0.0,
        "transcribe_run": {"started_epoch": None, "audio_seconds": 0.0},
        "created_at": config.created_at,
        "started_at": None,
        "finished_at": None,
        "updated_at": now_iso(),
        "run_started_epoch": None,
        "elapsed_before": 0.0,
        "errors": [],
        "warnings": [],
        "messages": [],
        "worker": None,
        "child_pids": [],
        "temp_paths": {
            "root": str(paths.root),
            "audio_dir": str(paths.audio_dir),
            "chunks_dir": str(paths.chunks_dir),
            "segments_dir": str(paths.segments_dir),
            "log_file": str(paths.log_file),
        },
        "output": {"dir": None, "files": {}},
        "resume_count": 0,
        "peak_rss_mb": 0.0,
        "last_writer": "ui",
    }


def load_config(task_id: str) -> TaskConfig:
    data = read_json(task_paths(task_id).config, default=None)
    if not isinstance(data, dict):
        raise CorruptedJSONError("O arquivo de configuração da tarefa está ausente ou corrompido.")
    return TaskConfig.from_dict(data)


def read_checkpoint(task_id: str) -> Optional[dict[str, Any]]:
    """Lê o checkpoint sem lock (a gravação é atômica). ``None`` se ausente/corrompido."""
    try:
        data = read_json(task_paths(task_id).checkpoint, default=None)
    except InvalidInputError:
        return None
    return data if isinstance(data, dict) else None


class CheckpointStore:
    """Único ponto de escrita do checkpoint (leitura-modificação-gravação sob lock)."""

    def __init__(self, task_id: str, *, writer: str) -> None:
        self.paths = task_paths(task_id)
        self.task_id = task_id
        self.writer = writer
        self._thread_lock = threading.RLock()
        self._last_throttled = 0.0

    def read(self) -> dict[str, Any]:
        data = read_checkpoint(self.task_id)
        if data is None:
            data = self._rebuild()
        return data

    def _rebuild(self) -> dict[str, Any]:
        """Reconstrói um checkpoint a partir de config.json + manifest.json (arquivo perdido/corrompido)."""
        logger.warning("Checkpoint ausente ou corrompido em %s; reconstruindo.", self.paths.checkpoint)
        config = load_config(self.task_id)
        data = initial_checkpoint(config)
        manifest = read_json(self.paths.manifest, default={}) or {}
        chunks = manifest.get("chunks") or []
        data["mode"] = manifest.get("mode")
        data["title"] = manifest.get("title") or data["title"]
        data["audio_duration"] = manifest.get("duration")
        data["total_blocks"] = len(chunks)
        data["completed_blocks"] = [c["index"] for c in chunks if c.get("status") == "done"]
        data["blocks"] = [
            {"index": c["index"], "start": c.get("start"), "end": c.get("end"), "status": c.get("status", "pending")}
            for c in chunks
        ]
        data["state"] = TaskState.FAILED.value
        data["message"] = "O progresso foi reconstruído a partir do disco. Retome a tarefa para continuar."
        data["warnings"].append("O arquivo de progresso estava corrompido e foi reconstruído.")
        return data

    def mutate(self, fn: Callable[[dict[str, Any]], None], *, timeout: float = 15.0) -> dict[str, Any]:
        """Aplica ``fn(checkpoint)`` e grava atomicamente. Retorna o novo estado."""
        with self._thread_lock:
            with FileLock(self.paths.checkpoint_lock, timeout=timeout):
                data = self.read()
                fn(data)
                data["updated_at"] = now_iso()
                data["last_writer"] = self.writer
                atomic_write_json(self.paths.checkpoint, data)
                return data

    def update(self, **changes: Any) -> dict[str, Any]:
        def _apply(ck: dict[str, Any]) -> None:
            ck.update(changes)

        return self.mutate(_apply)

    def mutate_throttled(self, fn: Callable[[dict[str, Any]], None], min_interval: float = 0.7) -> Optional[dict[str, Any]]:
        """Como ``mutate``, mas ignora chamadas muito frequentes (progresso intenso)."""
        now = time.monotonic()
        if now - self._last_throttled < min_interval:
            return None
        self._last_throttled = now
        return self.mutate(fn)


# -- Funções puras sobre o dicionário do checkpoint ---------------------------


def set_step(ck: dict[str, Any], step: str, status: Optional[str] = None, fraction: Optional[float] = None) -> None:
    entry = ck["steps"].setdefault(step, {"status": STEP_PENDING, "fraction": 0.0})
    if status is not None:
        entry["status"] = status
        if status == STEP_DONE:
            entry["fraction"] = 1.0
    if fraction is not None:
        entry["fraction"] = max(0.0, min(1.0, float(fraction)))


def compute_overall(ck: dict[str, Any]) -> float:
    """Progresso geral (0–1): média ponderada das frações reais das etapas não puladas."""
    total_weight = 0.0
    done_weight = 0.0
    for step, entry in ck.get("steps", {}).items():
        if entry.get("status") == STEP_SKIPPED:
            continue
        w = STEP_WEIGHTS.get(step, 0.0)
        total_weight += w
        done_weight += w * (1.0 if entry.get("status") == STEP_DONE else float(entry.get("fraction") or 0.0))
    return max(0.0, min(1.0, done_weight / total_weight)) if total_weight > 0 else 0.0


def add_message(ck: dict[str, Any], text: str, level: str = "info", keep: int = 40) -> None:
    ck.setdefault("messages", []).append({"time": now_iso(), "level": level, "text": text})
    del ck["messages"][:-keep]
    ck["message"] = text


def add_error(ck: dict[str, Any], exc: BaseException) -> None:
    entry = error_to_dict(exc)
    entry["state"] = ck.get("state")
    ck.setdefault("errors", []).append(entry)
    del ck["errors"][:-20]


def elapsed_seconds(ck: dict[str, Any], now: Optional[float] = None) -> float:
    """Tempo ativo de processamento (soma de todas as execuções, inclusive retomadas)."""
    now = now if now is not None else time.time()
    total = float(ck.get("elapsed_before") or 0.0)
    started = ck.get("run_started_epoch")
    if started:
        total += max(0.0, now - float(started))
    return total


def estimate_remaining_seconds(ck: dict[str, Any], now: Optional[float] = None) -> Optional[float]:
    """Tempo restante estimado com base na velocidade REAL medida (ou ``None`` se ainda não há medição)."""
    run = ck.get("transcribe_run") or {}
    started, processed = run.get("started_epoch"), float(run.get("audio_seconds") or 0.0)
    total = ck.get("audio_duration")
    if not started or not total or processed < 30:
        return None
    now = now if now is not None else time.time()
    speed = processed / max(1e-6, now - float(started))  # segundos de áudio por segundo real
    remaining_audio = max(0.0, float(total) - float(ck.get("transcribed_seconds") or 0.0))
    return remaining_audio / speed if speed > 0 else None


def current_state(ck: dict[str, Any], task_id: Optional[str] = None) -> TaskState:
    """Estado exibido: mostra ``cancel_requested`` se há pedido de cancelamento pendente."""
    state = TaskState(ck.get("state", TaskState.PENDING.value))
    if not state.is_terminal and state != TaskState.CANCEL_REQUESTED:
        tid = task_id or ck.get("task_id")
        if tid and is_cancel_requested(tid):
            return TaskState.CANCEL_REQUESTED
    return state


# =============================================================================
# Manifesto (escrito somente pelo worker)
# =============================================================================


def read_manifest(task_id: str) -> dict[str, Any]:
    data = read_json(task_paths(task_id).manifest, default={})
    return data if isinstance(data, dict) else {}


def write_manifest(task_id: str, manifest: dict[str, Any]) -> None:
    manifest["schema"] = SCHEMA_VERSION
    manifest["updated_at"] = now_iso()
    atomic_write_json(task_paths(task_id).manifest, manifest)


# =============================================================================
# Cancelamento (arquivo de sinalização)
# =============================================================================


def request_cancel(task_id: str, reason: str = "user") -> None:
    """Interface: grava ``cancel.json``. O worker o verifica em pontos seguros."""
    paths = task_paths(task_id)
    if not paths.root.is_dir():
        raise InvalidInputError("Tarefa não encontrada.")
    atomic_write_json(paths.cancel, {"requested_at": now_iso(), "by": "ui", "reason": reason}, keep_backup=False)
    # Se o worker já não existe, finaliza o cancelamento na hora.
    reconcile_task(task_id)


def is_cancel_requested(task_id: str) -> bool:
    try:
        return task_paths(task_id).cancel.exists()
    except InvalidInputError:
        return False


def clear_cancel(task_id: str) -> None:
    try:
        task_paths(task_id).cancel.unlink()
    except FileNotFoundError:
        pass


# =============================================================================
# Processos
# =============================================================================


def worker_alive(ck: Optional[dict[str, Any]]) -> bool:
    """O worker registrado está vivo? Compara PID + horário de criação + ID da tarefa na linha de comando."""
    if not ck:
        return False
    w = ck.get("worker") or {}
    return is_process_alive(w.get("pid"), w.get("create_time"), cmdline_marker=ck.get("task_id"))


def is_worker_alive(task_id: str) -> bool:
    return worker_alive(read_checkpoint(task_id))


def kill_orphans(ck: dict[str, Any]) -> list[int]:
    """Encerra FFmpeg/yt-dlp registrados no checkpoint que continuam rodando sem o worker."""
    killed: list[int] = []
    for child in ck.get("child_pids") or []:
        pid = child.get("pid")
        if pid and is_process_alive(pid, child.get("create_time")):
            killed += kill_process_tree(int(pid))
    return killed


def reconcile_task(task_id: str) -> Optional[dict[str, Any]]:
    """Detecta worker encerrado sem concluir e corrige o estado (interface).

    * Worker morto + estado não terminal + cancelamento pedido → ``cancelled``.
    * Worker morto + estado não terminal → ``failed`` (retomável), e limpa
      processos órfãos (FFmpeg/yt-dlp).
    * Tarefa ``pending`` sem worker nunca lançado → permanece ``pending``.
    """
    ck = read_checkpoint(task_id)
    if ck is None:
        return None
    state = TaskState(ck.get("state", TaskState.PENDING.value))
    if state.is_terminal or worker_alive(ck):
        return ck
    if state == TaskState.PENDING and not ck.get("worker"):
        return ck

    store = CheckpointStore(task_id, writer="ui")
    orphan_ck = ck

    def _apply(data: dict[str, Any]) -> None:
        # Reavalia dentro do lock: o worker pode ter terminado entre a leitura e agora.
        st = TaskState(data.get("state", TaskState.PENDING.value))
        if st.is_terminal or worker_alive(data):
            return
        if data.get("run_started_epoch"):
            data["elapsed_before"] = elapsed_seconds(data)
            data["run_started_epoch"] = None
        data["child_pids"] = []
        data["finished_at"] = now_iso()
        if is_cancel_requested(task_id):
            data["state"] = TaskState.CANCELLED.value
            add_message(data, "Tarefa cancelada. Os blocos já concluídos foram preservados.")
        else:
            exc = WorkerError(
                "O processamento em segundo plano foi encerrado antes de terminar (programa fechado, "
                "computador reiniciado ou erro inesperado). Os blocos já concluídos foram preservados: "
                "você pode retomar a tarefa.",
                technical=f"worker={data.get('worker')} estado={st.value}",
            )
            add_error(data, exc)
            data["state"] = TaskState.FAILED.value
            add_message(data, exc.user_message, "error")

    try:
        kill_orphans(orphan_ck)
        return store.mutate(_apply)
    except CheckpointError:
        logger.exception("Não foi possível reconciliar a tarefa %s", task_id)
        return ck


_launched: dict[str, subprocess.Popen] = {}  # evita ResourceWarning/zumbis do Popen


def _tail_text(path: Path, max_lines: int = 25) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def active_task_ids() -> list[str]:
    """IDs de tarefas com worker vivo e estado não terminal."""
    ids: list[str] = []
    if not TASKS_DIR.is_dir():
        return ids
    for d in TASKS_DIR.iterdir():
        if d.is_dir() and is_valid_task_id(d.name):
            ck = read_checkpoint(d.name)
            if ck and not TaskState(ck.get("state", "pending")).is_terminal and worker_alive(ck):
                ids.append(d.name)
    return ids


def start_worker(task_id: str, *, startup_timeout: float = 25.0, max_active: int = MAX_ACTIVE_WORKERS) -> int:
    """Inicia ``python -m src.worker --task-id ID`` como processo independente.

    Proteções: lock de lançamento, recusa se este worker já está vivo, limite de
    workers ativos, registro do PID + horário de criação no checkpoint e
    verificação de que o processo não morreu na inicialização.
    """
    paths = task_paths(task_id)
    if not paths.root.is_dir():
        raise WorkerError("A pasta da tarefa não existe.")

    with FileLock(paths.launch_lock, timeout=15):
        ck = read_checkpoint(task_id)
        if ck is None:
            raise WorkerError("Não foi possível ler o estado da tarefa.", technical="checkpoint ausente")
        if worker_alive(ck):
            raise WorkerError("Esta tarefa já está em execução.")
        if TaskState(ck.get("state", "pending")) == TaskState.COMPLETED:
            raise WorkerError("Esta tarefa já foi concluída.")
        others = [t for t in active_task_ids() if t != task_id]
        if len(others) >= max_active:
            raise WorkerError(
                "Já existe uma transcrição em andamento. Aguarde ela terminar ou cancele-a antes de iniciar outra."
            )

        paths.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.log_file, "ab") as log:
            log.write(f"\n===== Iniciando worker {task_id} em {now_iso()} =====\n".encode("utf-8"))
            log.flush()
            kwargs: dict[str, Any] = {}
            if os.name == "nt":
                # Grupo de processos próprio: Ctrl+C no console do Streamlit não derruba o worker.
                kwargs["creationflags"] = (
                    getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                )
            else:
                kwargs["start_new_session"] = True
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "src.worker", "--task-id", task_id],
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=subprocess_env(),
                    **kwargs,
                )
            except OSError as exc:
                raise WorkerError(technical=repr(exc)) from exc
        _launched[task_id] = proc

        store = CheckpointStore(task_id, writer="ui")
        identity = {
            "pid": proc.pid,
            "create_time": get_process_create_time(proc.pid),
            "launched_at": now_iso(),
            "started_at": None,
        }

        def _register(data: dict[str, Any]) -> None:
            data["worker"] = identity

        store.mutate(_register)

        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            code = proc.poll()
            if code is not None:
                tail = _tail_text(paths.log_file)
                reconcile_task(task_id)
                raise WorkerError(
                    "O processo de transcrição não conseguiu iniciar. Consulte o log da tarefa.",
                    technical=f"código de saída {code}\n{tail}",
                )
            ck = read_checkpoint(task_id) or {}
            if (ck.get("worker") or {}).get("started_at"):
                break
            time.sleep(0.2)
        return proc.pid


# =============================================================================
# Criação, retomada e listagem
# =============================================================================


def create_task(
    source_type: str,
    source: str,
    title: str,
    *,
    check_duplicates: bool = True,
    **options: Any,
) -> TaskConfig:
    """Cria a pasta, o ``config.json`` e o checkpoint inicial. Não inicia o worker."""
    if check_duplicates:
        for tid in active_task_ids():
            other = read_checkpoint(tid) or {}
            if other.get("source") == source:
                raise InvalidInputError("Esta mesma origem já está sendo transcrita.")

    config = TaskConfig(task_id=generate_task_id(), source_type=source_type, source=source, title=title, **options)
    config.validate()
    paths = task_paths(config.task_id)
    paths.root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(paths.config, config.to_dict())
    atomic_write_json(paths.checkpoint, initial_checkpoint(config))
    return config


def prepare_resume(task_id: str, *, device_override: Optional[str] = None) -> TaskConfig:
    """Interface: prepara a retomada (somente com o worker inativo)."""
    paths = task_paths(task_id)
    with FileLock(paths.launch_lock, timeout=15):
        ck = read_checkpoint(task_id)
        if ck is None:
            raise WorkerError("Não foi possível ler o estado da tarefa.")
        if worker_alive(ck):
            raise WorkerError("Esta tarefa ainda está em execução.")
        state = TaskState(ck.get("state", "pending"))
        if state == TaskState.COMPLETED:
            raise WorkerError("Esta tarefa já foi concluída.")
        kill_orphans(ck)
        config = load_config(task_id)
        if device_override:
            config.device = device_override
            config.allow_cpu_fallback = True
            config.validate()
            atomic_write_json(paths.config, config.to_dict())
        clear_cancel(task_id)

        def _apply(data: dict[str, Any]) -> None:
            data["state"] = TaskState.PENDING.value
            data["finished_at"] = None
            data["worker"] = None
            data["child_pids"] = []
            data["current_block"] = None
            data["resume_count"] = int(data.get("resume_count") or 0) + 1
            data["transcribe_run"] = {"started_epoch": None, "audio_seconds": 0.0}
            if device_override:
                data["device_requested"] = device_override
            add_message(data, "Retomando a tarefa a partir do último ponto salvo…")

        CheckpointStore(task_id, writer="ui").mutate(_apply)
        return config


def resume_task(task_id: str, *, device_override: Optional[str] = None) -> int:
    """Prepara e inicia o worker de uma tarefa interrompida. Retorna o PID."""
    prepare_resume(task_id, device_override=device_override)
    return start_worker(task_id)


@dataclass
class TaskSummary:
    task_id: str
    title: str
    source_type: str
    state: TaskState
    progress: float
    created_at: str
    model: str
    completed_blocks: int
    total_blocks: int
    alive: bool
    resumable: bool
    output_dir: Optional[str]
    elapsed: float


def summarize(ck: dict[str, Any]) -> TaskSummary:
    alive = worker_alive(ck)
    state = current_state(ck)
    return TaskSummary(
        task_id=ck["task_id"],
        title=ck.get("title") or ck["task_id"],
        source_type=ck.get("source_type", ""),
        state=state,
        progress=float(ck.get("overall_progress") or compute_overall(ck)),
        created_at=ck.get("created_at", ""),
        model=ck.get("model", ""),
        completed_blocks=len(ck.get("completed_blocks") or []),
        total_blocks=int(ck.get("total_blocks") or 0),
        alive=alive,
        resumable=(not alive) and TaskState(ck.get("state", "pending")) in RESUMABLE_STATES,
        output_dir=(ck.get("output") or {}).get("dir"),
        elapsed=elapsed_seconds(ck),
    )


def list_tasks(*, reconcile: bool = True) -> list[TaskSummary]:
    """Todas as tarefas, da mais recente para a mais antiga."""
    out: list[TaskSummary] = []
    if not TASKS_DIR.is_dir():
        return out
    for d in TASKS_DIR.iterdir():
        if not (d.is_dir() and is_valid_task_id(d.name)):
            continue
        ck = reconcile_task(d.name) if reconcile else read_checkpoint(d.name)
        if ck:
            out.append(summarize(ck))
    out.sort(key=lambda t: t.task_id, reverse=True)  # IDs começam com data/hora
    return out


def find_interrupted_tasks() -> list[TaskSummary]:
    """Tarefas que podem ser retomadas (interrompidas, canceladas ou com falha) e já têm progresso."""
    return [
        t for t in list_tasks()
        if t.resumable and t.state != TaskState.PENDING and (t.completed_blocks > 0 or t.state == TaskState.FAILED)
    ]


def delete_task(task_id: str) -> None:
    """Remove SOMENTE a pasta temporária da tarefa e o log dela. Nunca toca no arquivo original
    nem nos resultados em ``transcricoes/``."""
    paths = task_paths(task_id)
    ck = read_checkpoint(task_id)
    if worker_alive(ck):
        raise WorkerError("Cancele a tarefa antes de excluí-la.")
    root = paths.root.resolve()
    if root.parent != TASKS_DIR.resolve():
        raise InvalidInputError("Caminho de tarefa inesperado; exclusão recusada.")
    shutil.rmtree(root, ignore_errors=False)
    try:
        paths.log_file.unlink()
    except FileNotFoundError:
        pass
    _launched.pop(task_id, None)


def read_log_tail(task_id: str, max_lines: int = 60) -> str:
    return _tail_text(task_paths(task_id).log_file, max_lines)
