"""Testes de integração com o faster-whisper DE VERDADE (opcionais).

Como executar (baixa o modelo "base", ~150 MB, na primeira vez):

    set RUN_INTEGRATION=1
    python -m pytest tests/test_integration.py -s

Para validar também a qualidade do texto, aponte para um áudio curto com fala em português:

    set TRANSCRITOR_TEST_AUDIO=C:\\caminho\\fala_curta.wav
    set TRANSCRITOR_TEST_PHRASE=palavra1 palavra2        (opcional: palavras que devem aparecer)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import env_flag, wait_until

import src.task_manager as tm
import src.worker as wk

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not env_flag("RUN_INTEGRATION"), reason="defina RUN_INTEGRATION=1 para usar o modelo real"),
]


@pytest.fixture
def whisper():
    pytest.importorskip("faster_whisper")


def output_text(task_id) -> str:
    ck = tm.read_checkpoint(task_id)
    return Path(ck["output"]["files"]["txt"]).read_text(encoding="utf-8")


def test_short_audio_runs_the_whole_pipeline_with_the_real_model(whisper, data_dir, restore_signals, make_audio, tmp_path):
    src = make_audio(tmp_path / "curto.wav", [("tone", 4), ("silence", 2), ("tone", 4)])
    cfg = tm.create_task("local", str(src), "curto", model="base", language="pt", device="cpu", chunk_seconds=60)
    assert wk.TaskRunner(cfg.task_id).run() == wk.EXIT_OK
    ck = tm.read_checkpoint(cfg.task_id)
    assert ck["state"] == "completed" and ck["total_blocks"] == 1 and ck["device"]["device"] == "cpu"
    text = output_text(cfg.task_id)
    assert "TRANSCRIÇÃO" in text and "Modelo: base" in text
    for ext in ("txt", "srt", "vtt"):
        assert Path(ck["output"]["files"][ext]).is_file()
    print(f"\n[integração] pico de RAM do worker: {ck['peak_rss_mb']} MB; "
          f"tempo: {ck['elapsed_before']:.1f} s para 10 s de áudio")


@pytest.mark.skipif(not os.environ.get("TRANSCRITOR_TEST_AUDIO"), reason="defina TRANSCRITOR_TEST_AUDIO com um áudio de fala")
def test_real_speech_is_transcribed(whisper, data_dir, restore_signals):
    audio = Path(os.environ["TRANSCRITOR_TEST_AUDIO"])
    cfg = tm.create_task("local", str(audio), audio.stem, model="base", language="pt", device="cpu",
                         chunk_seconds=60, timestamp_mode="segment")
    assert wk.TaskRunner(cfg.task_id).run() == wk.EXIT_OK
    text = output_text(cfg.task_id).split("=" * 72)[-1].lower()
    assert "nenhuma fala foi detectada" not in text and len(text.split()) >= 3
    phrase = os.environ.get("TRANSCRITOR_TEST_PHRASE", "").lower().split()
    missing = [w for w in phrase if w not in text]
    assert not missing, f"palavras esperadas ausentes: {missing}"


def test_end_to_end_through_the_independent_worker_process(whisper, data_dir, make_audio, tmp_path):
    """Interface → Popen(python -m src.worker) → checkpoint JSON, exatamente como o app faz."""
    src = make_audio(tmp_path / "curto2.wav", [("tone", 3), ("silence", 2), ("tone", 3)])
    cfg = tm.create_task("local", str(src), "curto2", model="base", language="pt", device="cpu", chunk_seconds=60)
    tm.start_worker(cfg.task_id)
    assert wait_until(lambda: tm.TaskState(tm.read_checkpoint(cfg.task_id)["state"]).is_terminal, 900, 1.0)
    ck = tm.read_checkpoint(cfg.task_id)
    assert ck["state"] == "completed", ck["errors"]
    assert not tm.is_worker_alive(cfg.task_id)
