"""Fixtures e utilitários compartilhados pelos testes.

* ``data_dir``: isola ``temp/``, ``logs/`` e ``transcricoes/`` em uma pasta temporária,
  tanto para o processo de teste quanto para workers lançados como subprocessos
  (a variável ``TRANSCRITOR_DATA_DIR`` é herdada).
* ``ffmpeg``: pula o teste quando o FFmpeg não está instalado.
* ``make_audio``: gera áudio sintético (tons e silêncios) com o próprio FFmpeg.
* ``spawn_sleeper``: cria processos "dorminhocos" para simular workers/filhos vivos.
"""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Isolamento de pastas
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Redireciona temp/, logs/ e transcricoes/ para ``tmp_path`` (inclusive em subprocessos)."""
    import src
    import src.task_manager as tm
    import src.worker as wk

    tasks = tmp_path / "temp" / "tasks"
    logs = tmp_path / "logs"
    out = tmp_path / "transcricoes"
    models = tmp_path / "models"
    for d in (tasks, logs, out, models):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("TRANSCRITOR_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(src, "TASKS_DIR", tasks)
    monkeypatch.setattr(src, "LOGS_DIR", logs)
    monkeypatch.setattr(src, "TRANSCRIPTIONS_DIR", out)
    monkeypatch.setattr(tm, "TASKS_DIR", tasks)
    monkeypatch.setattr(tm, "LOGS_DIR", logs)
    monkeypatch.setattr(wk, "TRANSCRIPTIONS_DIR", out)
    return tmp_path


@pytest.fixture
def restore_signals():
    """O worker instala handlers de sinal; restaura os do pytest ao final."""
    names = [n for n in ("SIGINT", "SIGTERM", "SIGBREAK") if hasattr(signal, n)]
    saved = {n: signal.getsignal(getattr(signal, n)) for n in names}
    yield
    for n, handler in saved.items():
        try:
            signal.signal(getattr(signal, n), handler)
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# FFmpeg e áudio sintético
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ffmpeg():
    from src.audio_processor import find_ffmpeg
    from src.utils import FFmpegNotFoundError

    try:
        return find_ffmpeg()
    except FFmpegNotFoundError:
        pytest.skip("FFmpeg não está instalado/acessível no PATH")


def _make_audio(
    ffmpeg_exe: str,
    path: Path,
    parts: Sequence[tuple[str, float]],
    *,
    rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
) -> Path:
    """Gera áudio concatenando ``("tone"|"silence", segundos)``. Tons ficam bem acima de -35 dB."""
    args = [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y"]
    filters = []
    for i, (kind, dur) in enumerate(parts):
        if kind == "tone":
            args += ["-f", "lavfi", "-i", f"sine=frequency={440 + 10 * i}:sample_rate={rate}:duration={dur}"]
        else:
            args += ["-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=mono:d={dur}"]
        filters.append(f"[{i}:a]aformat=sample_rates={rate}:channel_layouts=mono[a{i}]")
    n = len(parts)
    graph = ";".join(filters) + ";" + "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[out]"
    args += ["-filter_complex", graph, "-map", "[out]", "-ac", str(channels), "-ar", str(rate), "-c:a", codec, str(path)]
    subprocess.run(args, check=True, capture_output=True)
    return path


@pytest.fixture
def make_audio(ffmpeg) -> Callable[..., Path]:
    def _factory(path: Path, parts, **kw) -> Path:
        return _make_audio(ffmpeg.ffmpeg, Path(path), parts, **kw)

    return _factory


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Processos auxiliares
# ---------------------------------------------------------------------------


@pytest.fixture
def spawn_sleeper():
    """Cria processos Python que dormem; ``marker`` entra na linha de comando (simula o ID da tarefa)."""
    procs: list[subprocess.Popen] = []

    def _spawn(marker: str = "", seconds: int = 120) -> subprocess.Popen:
        p = subprocess.Popen(
            [sys.executable, "-c", "import time, sys; time.sleep(int(sys.argv[1]))", str(seconds), marker],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(p)
        time.sleep(0.3)  # deixa o processo registrar sua linha de comando
        return p

    yield _spawn
    for p in procs:
        if p.poll() is None:
            p.kill()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


@pytest.fixture
def dead_pid() -> int:
    """PID de um processo que acabou de terminar (praticamente impossível de ter sido reutilizado)."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def wait_until(predicate: Callable[[], bool], timeout: float = 30.0, interval: float = 0.1) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "yes")
