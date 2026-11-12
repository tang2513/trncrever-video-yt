"""Transcritor de Vídeos Longos.

Pacote com os módulos de processamento (sem dependência do Streamlit):

- ``utils``           : caminhos, erros amigáveis, JSON atômico, locks, processos.
- ``hardware``        : detecção de CPU/RAM/GPU e escolha do dispositivo.
- ``subtitles``       : modelo de segmento, leitura VTT/SRT, escolha de legendas
                        e geração de TXT/SRT/VTT.
- ``downloader``      : validação de links e uso do yt-dlp (YouTube/Twitch).
- ``audio_processor`` : FFmpeg (verificação, extração, silencedetect) e divisão
                        do áudio em blocos com continuidade exata.
- ``transcription``   : (Parte 2) faster-whisper por bloco.
- ``task_manager``    : (Parte 2) criação/retomada/cancelamento de tarefas.
- ``worker``          : (Parte 2) processo independente executado via
                        ``python -m src.worker --task-id ID``.

Nenhum módulo deste pacote importa ``app.py`` ou o Streamlit, para que o
worker funcione como processo totalmente independente da interface.
"""

import os
from pathlib import Path

__version__ = "1.0.0"
APP_NAME = "Transcritor de Vídeos Longos"

#: Raiz do projeto (pasta que contém ``app.py``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Raiz dos dados. Normalmente é a própria pasta do projeto; a variável de ambiente
#: ``TRANSCRITOR_DATA_DIR`` permite mover ``transcricoes/``, ``temp/``, ``logs/`` e
#: ``models/`` para outro lugar (usado pelos testes para não tocar nos dados reais;
#: o worker, por ser um processo filho, herda a variável).
DATA_ROOT: Path = Path(os.environ.get("TRANSCRITOR_DATA_DIR") or PROJECT_ROOT).resolve()

#: Pastas de trabalho usadas por todos os módulos.
TRANSCRIPTIONS_DIR: Path = DATA_ROOT / "transcricoes"
TEMP_DIR: Path = DATA_ROOT / "temp"
TASKS_DIR: Path = TEMP_DIR / "tasks"
LOGS_DIR: Path = DATA_ROOT / "logs"
MODELS_DIR: Path = DATA_ROOT / "models"


def ensure_project_dirs() -> None:
    """Cria as pastas de trabalho, caso ainda não existam."""
    for folder in (TRANSCRIPTIONS_DIR, TASKS_DIR, LOGS_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


__all__ = [
    "__version__",
    "APP_NAME",
    "PROJECT_ROOT",
    "TRANSCRIPTIONS_DIR",
    "TEMP_DIR",
    "TASKS_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "ensure_project_dirs",
]
