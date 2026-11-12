"""Testes do worker (src/worker.py): pipeline completo, retomada, cancelamento e erros.

O faster-whisper é substituído por um motor simulado, de modo que o restante do pipeline
(FFmpeg real, blocos, manifesto, checkpoint, exportação) é exercitado de verdade.
Testes que dependem do FFmpeg são pulados quando ele não está instalado.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import ROOT, sha256

import src.audio_processor as ap
import src.task_manager as tm
import src.utils as u
import src.worker as wk
from src.audio_processor import ChunkSpec, FFmpegInfo
from src.hardware import DeviceDecision
from src.subtitles import Segment, load_segments_json
from src.transcription import ChunkTranscription

ROLLING_VTT = (
    "WEBVTT\n\n"
    "00:00:00.160 --> 00:00:02.230 align:start position:0%\n \n"
    "olá<00:00:00.640><c> pessoal</c><00:00:01.040><c> tudo</c><00:00:01.439><c> bem</c>\n\n"
    "00:00:02.230 --> 00:00:02.240 align:start position:0%\nolá pessoal tudo bem\n \n\n"
    "00:00:02.240 --> 00:00:04.150 align:start position:0%\nolá pessoal tudo bem\n"
    "hoje<00:00:02.800><c> vamos</c><00:00:03.200><c> falar</c>\n\n"
    "00:00:04.150 --> 00:00:04.160 align:start position:0%\nhoje vamos falar\n \n\n"
    "00:00:04.160 --> 00:00:06.000 align:start position:0%\nhoje vamos falar\n"
    "sobre<00:00:04.500><c> testes</c>\n"
)
DUMMY_FFMPEG = FFmpegInfo("ffmpeg", "ffprobe", "teste")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake(data_dir, monkeypatch, restore_signals):
    """Substitui o faster-whisper e a detecção de hardware; devolve o objeto de controle."""
    ctl = SimpleNamespace(calls=[], fail_on=None, on_block=None, sleep=0.0, instances=[])

    class FakeEngine:
        def __init__(self, settings, decision, **kw):
            self.settings, self.decision, self.kw = settings, decision, kw
            ctl.instances.append(self)

        def load(self, cancel_check=None):
            pass

        def transcribe_file(self, wav_path, *, offset, chunk_duration, language=None, progress_cb=None, cancel_check=None):
            idx = int(Path(wav_path).stem.split("_")[1])
            ctl.calls.append(idx)
            if ctl.on_block:
                ctl.on_block(idx)
            if cancel_check and cancel_check():
                raise u.TaskCancelledError()
            if ctl.fail_on == idx:
                raise RuntimeError("falha simulada no bloco")
            time.sleep(ctl.sleep)
            if progress_cb:
                progress_cb(chunk_duration / 2, chunk_duration)
                progress_cb(chunk_duration, chunk_duration)
            return ChunkTranscription(
                segments=[Segment(offset + 1.0, offset + 3.0, f"bloco {idx + 1}")],
                language="pt", language_probability=0.99, chunk_duration=chunk_duration, elapsed=0.01)

        def close(self):
            pass

    monkeypatch.setattr(wk, "TranscriptionEngine", FakeEngine)
    monkeypatch.setattr(wk, "detect_hardware", lambda: object())
    monkeypatch.setattr(wk, "resolve_device",
                        lambda requested, model, hw, allow_cpu_fallback=True: DeviceDecision("cpu", "int8", 2))
    ctl.dummy_ffmpeg = lambda: monkeypatch.setattr(wk, "find_ffmpeg", lambda: DUMMY_FFMPEG)
    return ctl


def audio_task(make_audio, tmp_path, *, seconds=200, name="palestra.wav", **options):
    (tmp_path / "fonte").mkdir(exist_ok=True)
    src = make_audio(tmp_path / "fonte" / name, [("tone", seconds)])
    opts = dict(chunk_seconds=60, use_silence_split=False, model="base", timestamp_mode="segment")
    opts.update(options)
    cfg = tm.create_task("local", str(src), Path(name).stem, **opts)
    return cfg.task_id, src


def run(task_id) -> int:
    return wk.TaskRunner(task_id).run()


def ck_of(task_id):
    return tm.read_checkpoint(task_id)


def manifest_chunks(task_id):
    return [ChunkSpec.from_dict(c) for c in tm.read_manifest(task_id)["chunks"]]


def assert_timeline_ok(task_id):
    """Ordem dos blocos, continuidade, ausência de lacunas/duplicações e tempos dentro de cada bloco."""
    m = tm.read_manifest(task_id)
    chunks = manifest_chunks(task_id)
    ap.verify_chunk_continuity(chunks, m["wav"]["frames"])
    previous_start = -1.0
    for spec, entry in zip(chunks, m["chunks"]):
        assert entry["status"] == "done"
        segs = load_segments_json(entry["segments_file"])
        for s in segs:
            assert spec.start - 1e-6 <= s.start <= s.end <= spec.end + 1e-6
            assert s.start >= previous_start
            previous_start = s.start


# ---------------------------------------------------------------------------
# Pipeline completo (FFmpeg real + motor simulado)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_complete_run(self, fake, make_audio, tmp_path, data_dir):
        tid, src = audio_task(make_audio, tmp_path)
        before = sha256(src)

        assert run(tid) == wk.EXIT_OK
        ck = ck_of(tid)

        assert ck["state"] == "completed" and ck["overall_progress"] == 1.0 and ck["mode"] == "audio"
        assert ck["total_blocks"] == 3 and ck["completed_blocks"] == [0, 1, 2] and ck["current_block"] is None
        assert [b["status"] for b in ck["blocks"]] == ["done"] * 3
        assert ck["transcribed_seconds"] == pytest.approx(ck["audio_duration"], abs=0.01)
        for step in ("validate", "convert", "split", "load_model", "transcribe", "organize", "finish"):
            assert ck["steps"][step]["status"] == "done", step
        assert ck["steps"]["subtitles"]["status"] == ck["steps"]["download"]["status"] == "skipped"
        assert ck["language_detected"] == "pt" and ck["device"]["label"] == "CPU · int8"
        assert ck["finished_at"] and ck["run_started_epoch"] is None and ck["elapsed_before"] > 0
        assert ck["peak_rss_mb"] > 0 and ck["child_pids"] == [] and ck["errors"] == []

        # resultados
        out = Path(ck["output"]["dir"])
        assert out == data_dir / "transcricoes" / "palestra"
        txt = (out / "palestra.txt").read_text(encoding="utf-8")
        assert "TRANSCRIÇÃO" in txt and "Modelo: base" in txt and "Idioma: Português" in txt
        assert "[00:00:01] bloco 1\n" in txt and "[00:01:01] bloco 2\n" in txt and "[00:02:01] bloco 3\n" in txt
        srt = (out / "palestra.srt").read_text(encoding="utf-8")
        assert "00:00:01,000 --> 00:00:03,000" in srt and "00:01:01,000 --> 00:01:03,000" in srt
        assert "00:02:01,000 --> 00:02:03,000" in srt  # tempo do vídeo original, não do processamento
        assert (out / "palestra.vtt").read_text(encoding="utf-8").startswith("WEBVTT")

        # blocos e manifesto
        assert_timeline_ok(tid)
        assert fake.calls == [0, 1, 2]

        # limpeza e segurança
        paths = tm.task_paths(tid)
        assert not paths.audio_dir.exists() and not paths.chunks_dir.exists() and not paths.work_dir.exists()
        assert paths.segments_dir.is_dir() and not paths.worker_lock.exists()
        assert src.is_file() and sha256(src) == before  # o original nunca é alterado

    def test_ui_can_follow_progress_through_the_json_file(self, fake, make_audio, tmp_path):
        fake.sleep = 0.4
        tid, _ = audio_task(make_audio, tmp_path)
        seen: list[dict] = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                ck = tm.read_checkpoint(tid)
                if ck is not None:
                    seen.append({"state": ck["state"], "p": ck["overall_progress"], "n": len(ck["completed_blocks"])})
                time.sleep(0.02)

        t = threading.Thread(target=reader)
        t.start()
        try:
            assert run(tid) == wk.EXIT_OK
        finally:
            time.sleep(0.2)
            stop.set()
            t.join()

        states = {s["state"] for s in seen}
        assert {"transcribing", "completed"} <= states
        progress = [s["p"] for s in seen]
        assert all(b >= a - 1e-9 for a, b in zip(progress, progress[1:])), "o progresso nunca pode voltar atrás"
        blocks = [s["n"] for s in seen]
        assert blocks == sorted(blocks) and max(blocks) == 3
        assert 0.0 <= min(progress) and max(progress) == 1.0

    def test_silence_based_cuts_are_stored_in_the_manifest(self, fake, make_audio, ffmpeg, tmp_path):
        (tmp_path / "fonte").mkdir()
        src = make_audio(tmp_path / "fonte" / "p.wav", [("tone", 70), ("silence", 1), ("tone", 70), ("silence", 1), ("tone", 40)])
        cfg = tm.create_task("local", str(src), "p", chunk_seconds=60, model="base", silence={"search_window": 20})
        assert run(cfg.task_id) == wk.EXIT_OK
        chunks = manifest_chunks(cfg.task_id)
        assert [c.cut_reason for c in chunks[:-1]].count("silence") >= 1
        assert_timeline_ok(cfg.task_id)

    def test_cli_entrypoint_is_independent_of_the_interface(self, data_dir):
        kw = dict(cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        r = subprocess.run([sys.executable, "-m", "src.worker", "--task-id", "invalido"], **kw)
        assert r.returncode == wk.EXIT_BAD_ARGS
        r = subprocess.run([sys.executable, "-m", "src.worker", "--task-id", "20240101-000000-abcdef12"], **kw)
        assert r.returncode == wk.EXIT_BAD_ARGS  # tarefa inexistente
        r = subprocess.run([sys.executable, "-c",
                            "import sys, src.worker; print('streamlit' in sys.modules, 'app' in sys.modules)"], **kw)
        assert r.stdout.split()[-2:] == ["False", "False"]  # o worker não depende da interface


# ---------------------------------------------------------------------------
# 12. Retomada
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_continues_from_the_last_completed_block(self, fake, make_audio, tmp_path):
        tid, _ = audio_task(make_audio, tmp_path)
        fake.fail_on = 1
        assert run(tid) == wk.EXIT_FAILED
        ck = ck_of(tid)
        paths = tm.task_paths(tid)
        assert ck["state"] == "failed" and ck["completed_blocks"] == [0] and ck["errors"][-1]["code"] == "unexpected"
        assert "RuntimeError" in ck["errors"][-1]["message"]
        assert paths.segments_file(0).is_file() and not paths.segments_file(1).exists()
        done_mtime = paths.segments_file(0).stat().st_mtime_ns
        assert tm.summarize(ck).resumable

        fake.fail_on = None
        fake.calls.clear()
        tm.prepare_resume(tid)
        assert run(tid) == wk.EXIT_OK
        assert fake.calls == [1, 2]  # o bloco 0 NÃO foi refeito
        assert paths.segments_file(0).stat().st_mtime_ns == done_mtime
        ck = ck_of(tid)
        assert ck["state"] == "completed" and ck["completed_blocks"] == [0, 1, 2] and ck["resume_count"] == 1
        srt = Path(ck["output"]["files"]["srt"]).read_text(encoding="utf-8")
        assert srt.count("-->") == 3
        assert_timeline_ok(tid)

    def test_resume_recreates_a_deleted_chunk_file(self, fake, make_audio, tmp_path):
        tid, _ = audio_task(make_audio, tmp_path)
        fake.fail_on = 1
        assert run(tid) == wk.EXIT_FAILED
        paths = tm.task_paths(tid)
        (paths.chunks_dir / ap.chunk_filename(2)).unlink()  # o usuário limpou parte da pasta temporária
        fake.fail_on = None
        fake.calls.clear()
        tm.prepare_resume(tid)
        assert run(tid) == wk.EXIT_OK
        assert fake.calls == [1, 2] and ck_of(tid)["state"] == "completed"
        assert_timeline_ok(tid)

    def test_split_points_are_reused_not_recomputed(self, fake, make_audio, tmp_path):
        tid, _ = audio_task(make_audio, tmp_path)
        fake.fail_on = 0
        assert run(tid) == wk.EXIT_FAILED
        plan = [c.to_dict() for c in manifest_chunks(tid)]
        fake.fail_on = None
        tm.prepare_resume(tid)
        assert run(tid) == wk.EXIT_OK
        assert [c.to_dict() for c in manifest_chunks(tid)] == plan

    def test_resume_after_the_worker_was_killed(self, fake, make_audio, tmp_path, dead_pid):
        """Simula queda do computador: o checkpoint ficou em 'transcribing' com um worker que não existe mais."""
        tid, _ = audio_task(make_audio, tmp_path)
        fake.fail_on = 1
        run(tid)
        tm.CheckpointStore(tid, writer="test").update(
            state="transcribing", worker={"pid": dead_pid, "create_time": 1.0}, finished_at=None)
        assert tm.reconcile_task(tid)["state"] == "failed"
        fake.fail_on = None
        fake.calls.clear()
        tm.prepare_resume(tid)
        assert run(tid) == wk.EXIT_OK and fake.calls == [1, 2]

    def test_second_run_of_a_completed_task_is_refused_by_the_task_manager(self, fake, make_audio, tmp_path):
        tid, _ = audio_task(make_audio, tmp_path)
        assert run(tid) == wk.EXIT_OK
        with pytest.raises(u.WorkerError):
            tm.prepare_resume(tid)


# ---------------------------------------------------------------------------
# 13. Cancelamento
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_file_stops_at_a_safe_point_and_keeps_completed_blocks(self, fake, make_audio, tmp_path, data_dir):
        tid, _ = audio_task(make_audio, tmp_path)
        paths = tm.task_paths(tid)

        def on_block(idx):
            if idx == 1:  # a interface pede o cancelamento durante o bloco 2
                u.atomic_write_json(paths.cancel, {"requested_at": u.now_iso(), "by": "ui"}, keep_backup=False)

        fake.on_block = on_block
        assert run(tid) == wk.EXIT_CANCELLED
        ck = ck_of(tid)
        assert ck["state"] == "cancelled" and ck["completed_blocks"] == [0] and ck["current_block"] is None
        assert any("Cancelamento solicitado" in m["text"] for m in ck["messages"])
        assert paths.segments_file(0).is_file() and not paths.segments_file(1).exists()
        assert ck["child_pids"] == [] and not paths.worker_lock.exists()
        assert not (data_dir / "transcricoes" / "palestra").exists()  # nada parcial é publicado

        # retomar depois do cancelamento
        fake.on_block = None
        fake.calls.clear()
        tm.prepare_resume(tid)
        assert not paths.cancel.exists()
        assert run(tid) == wk.EXIT_OK and fake.calls == [1, 2]

    def test_system_signal_is_handled_like_a_cancel(self, fake, make_audio, tmp_path):
        fake.dummy_ffmpeg()
        cfg = tm.create_task("local", str(tmp_path / "x.mp3"), "x")
        (tmp_path / "x.mp3").write_bytes(b"abc")
        runner = wk.TaskRunner(cfg.task_id)
        runner._signal_received = True
        assert runner.run() == wk.EXIT_CANCELLED
        ck = ck_of(cfg.task_id)
        assert ck["state"] == "cancelled" and "sinal do sistema" in ck["message"]

    def test_cancel_requested_before_any_work_leaves_nothing_behind(self, fake, make_audio, tmp_path):
        tid, src = audio_task(make_audio, tmp_path, seconds=30)
        paths = tm.task_paths(tid)
        u.atomic_write_json(paths.cancel, {"requested_at": u.now_iso()}, keep_backup=False)
        assert run(tid) == wk.EXIT_CANCELLED
        assert not paths.audio_dir.exists() or not list(paths.audio_dir.glob("*.wav"))
        ck = ck_of(tid)
        assert ck["state"] == "cancelled" and ck["child_pids"] == [] and fake.calls == [] and src.is_file()


# ---------------------------------------------------------------------------
# 18. Tratamento de erros (mensagens compreensíveis + estado 'failed')
# ---------------------------------------------------------------------------


class TestErrors:
    def failed(self, tid):
        ck = ck_of(tid)
        assert ck["state"] == "failed" and ck["finished_at"] and ck["child_pids"] == []
        return ck["errors"][-1]

    def test_missing_file(self, fake, tmp_path):
        fake.dummy_ffmpeg()
        cfg = tm.create_task("local", str(tmp_path / "nao_existe.mp4"), "x")
        assert run(cfg.task_id) == wk.EXIT_FAILED
        err = self.failed(cfg.task_id)
        assert err["code"] == "invalid_input" and err["message"].startswith("Arquivo não encontrado")

    def test_unsupported_format(self, fake, tmp_path):
        fake.dummy_ffmpeg()
        f = tmp_path / "texto.txt"
        f.write_text("oi")
        cfg = tm.create_task("local", str(f), "texto")
        assert run(cfg.task_id) == wk.EXIT_FAILED
        assert self.failed(cfg.task_id)["code"] == "unsupported_format"

    def test_empty_file(self, fake, tmp_path):
        fake.dummy_ffmpeg()
        f = tmp_path / "vazio.mp3"
        f.write_bytes(b"")
        cfg = tm.create_task("local", str(f), "vazio")
        assert run(cfg.task_id) == wk.EXIT_FAILED
        assert self.failed(cfg.task_id)["code"] == "corrupted_media"

    def test_corrupted_media(self, fake, ffmpeg, tmp_path):
        f = tmp_path / "ruim.mp3"
        f.write_bytes(bytes(range(256)) * 64)
        cfg = tm.create_task("local", str(f), "ruim")
        assert run(cfg.task_id) == wk.EXIT_FAILED
        assert self.failed(cfg.task_id)["code"] == "corrupted_media"

    def test_ffmpeg_not_installed(self, fake, monkeypatch, tmp_path):
        def missing():
            raise u.FFmpegNotFoundError()

        monkeypatch.setattr(wk, "find_ffmpeg", missing)
        f = tmp_path / "a.mp3"
        f.write_bytes(b"abc")
        cfg = tm.create_task("local", str(f), "a")
        assert run(cfg.task_id) == wk.EXIT_FAILED
        err = self.failed(cfg.task_id)
        assert err["code"] == "ffmpeg_not_found" and "PATH" in err["message"]

    def test_not_enough_disk_space(self, fake, make_audio, monkeypatch, tmp_path):
        def full(*a, **k):
            raise u.DiskSpaceError()

        monkeypatch.setattr(ap, "ensure_disk_space", full)
        tid, _ = audio_task(make_audio, tmp_path, seconds=20)
        assert run(tid) == wk.EXIT_FAILED
        assert self.failed(tid)["code"] == "disk_space"

    def test_gpu_unavailable_then_resume_on_cpu(self, fake, make_audio, monkeypatch, tmp_path):
        def no_gpu(requested, model, hw, allow_cpu_fallback=True):
            raise u.GPUError("A GPU/CUDA não está disponível: sem driver. Deseja processar pela CPU?")

        monkeypatch.setattr(wk, "resolve_device", no_gpu)
        tid, _ = audio_task(make_audio, tmp_path, device="cuda", allow_cpu_fallback=False)
        assert run(tid) == wk.EXIT_FAILED
        assert self.failed(tid)["code"] == "gpu"
        assert ck_of(tid)["completed_blocks"] == []

        monkeypatch.setattr(wk, "resolve_device",
                            lambda requested, model, hw, allow_cpu_fallback=True: DeviceDecision("cpu", "int8", 2))
        tm.prepare_resume(tid, device_override="cpu")
        assert run(tid) == wk.EXIT_OK
        assert ck_of(tid)["state"] == "completed"

    def test_unexpected_exception_is_logged_and_reported_without_traceback_in_the_message(self, fake, make_audio, tmp_path):
        tid, _ = audio_task(make_audio, tmp_path, seconds=70)
        fake.fail_on = 0
        run(tid)
        err = ck_of(tid)["errors"][-1]
        assert "Traceback" not in err["message"] and "falha simulada" in err["technical"]

    def test_two_workers_cannot_run_the_same_task(self, fake, tmp_path):
        fake.dummy_ffmpeg()
        f = tmp_path / "a.mp3"
        f.write_bytes(b"abc")
        cfg = tm.create_task("local", str(f), "a")
        with u.FileLock(tm.task_paths(cfg.task_id).worker_lock, stale_after=10 ** 9):
            assert run(cfg.task_id) == wk.EXIT_ALREADY_RUNNING
        assert ck_of(cfg.task_id)["state"] == "pending"  # o segundo worker não mexeu em nada


# ---------------------------------------------------------------------------
# YouTube: legendas e alternativa por áudio (yt-dlp simulado)
# ---------------------------------------------------------------------------

YT_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def patch_ytdlp(monkeypatch, *, subtitles: bool, subtitle_fails: bool = False, wav: Path | None = None):
    meta = {
        "title": "Vídeo de Teste", "duration": 100.0, "language": "pt", "is_live": False,
        "subtitles": {"pt-BR": [{"ext": "vtt", "name": "Português"}]} if subtitles else {},
        "automatic_captions": {},
    }
    monkeypatch.setattr(wk, "check_ytdlp", lambda: "2025.1.1")
    monkeypatch.setattr(wk, "prepare_cookies_copy", lambda *a, **k: None)
    monkeypatch.setattr(wk, "fetch_metadata", lambda url, **kw: meta)
    downloads = SimpleNamespace(subtitle=0, audio=0)

    def fake_subtitle(url, track, out_dir, **kw):
        downloads.subtitle += 1
        if subtitle_fails:
            raise u.ContentUnavailableError()
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"subtitle.{track.lang}.vtt"
        p.write_text(ROLLING_VTT, encoding="utf-8")
        return p

    def fake_audio(url, out_dir, **kw):
        downloads.audio += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "source_audio.wav"
        shutil.copy(wav, dest)
        return dest

    monkeypatch.setattr(wk, "download_subtitle", fake_subtitle)
    monkeypatch.setattr(wk, "download_audio", fake_audio)
    return downloads


class TestYouTube:
    def test_uses_portuguese_subtitles_without_downloading_audio_or_transcribing(self, fake, monkeypatch, data_dir):
        fake.dummy_ffmpeg()
        downloads = patch_ytdlp(monkeypatch, subtitles=True)
        cfg = tm.create_task("youtube", YT_URL, YT_URL, timestamp_mode="segment")
        assert run(cfg.task_id) == wk.EXIT_OK
        ck = ck_of(cfg.task_id)
        assert ck["state"] == "completed" and ck["mode"] == "subtitles" and ck["title"] == "Vídeo de Teste"
        assert downloads.subtitle == 1 and downloads.audio == 0 and fake.calls == [] and fake.instances == []
        for step in ("download", "convert", "split", "load_model", "transcribe"):
            assert ck["steps"][step]["status"] == "skipped", step
        assert ck["steps"]["subtitles"]["status"] == "done" and ck["language_detected"] == "pt"
        out = Path(ck["output"]["dir"])
        assert out == data_dir / "transcricoes" / "Vídeo de Teste"
        txt = (out / "Vídeo de Teste.txt").read_text(encoding="utf-8")
        assert "Legendas do YouTube (adicionada pelo criador, idioma pt-BR)" in txt
        assert "[00:00:00] olá pessoal tudo bem" in txt and "[00:00:02] hoje vamos falar" in txt
        assert (out / "Vídeo de Teste.srt").read_text(encoding="utf-8").count("-->") == 3
        assert not tm.task_paths(cfg.task_id).work_dir.exists()

    @pytest.mark.parametrize("has_subtitles,subtitle_fails,expected", [
        (False, False, "não possui legendas em português"),
        (True, True, "Não foi possível usar a legenda"),
    ])
    def test_falls_back_to_audio_transcription(self, fake, make_audio, monkeypatch, tmp_path, has_subtitles,
                                               subtitle_fails, expected):
        wav = make_audio(tmp_path / "baixado.wav", [("tone", 100)])
        downloads = patch_ytdlp(monkeypatch, subtitles=has_subtitles, subtitle_fails=subtitle_fails, wav=wav)
        cfg = tm.create_task("youtube", YT_URL, YT_URL, chunk_seconds=60, use_silence_split=False, model="base")
        assert run(cfg.task_id) == wk.EXIT_OK
        ck = ck_of(cfg.task_id)
        assert ck["mode"] == "audio" and downloads.audio == 1 and fake.calls == [0, 1]
        assert any(expected in m["text"] for m in ck["messages"])
        assert ck["steps"]["download"]["status"] == "done" and ck["steps"]["subtitles"]["status"] == "done"
        assert Path(ck["output"]["dir"]).name == "Vídeo de Teste"
        assert not any(tm.task_paths(cfg.task_id).audio_dir.glob("*")) if tm.task_paths(cfg.task_id).audio_dir.exists() else True
        assert_timeline_ok(cfg.task_id)

    def test_ignore_subtitles_mode_transcribes_the_audio(self, fake, make_audio, monkeypatch, tmp_path):
        wav = make_audio(tmp_path / "baixado.wav", [("tone", 100)])
        downloads = patch_ytdlp(monkeypatch, subtitles=True, wav=wav)
        cfg = tm.create_task("youtube", YT_URL, YT_URL, subtitle_mode="ignore", chunk_seconds=60,
                             use_silence_split=False, model="base")
        assert run(cfg.task_id) == wk.EXIT_OK
        assert downloads.subtitle == 0 and downloads.audio == 1 and ck_of(cfg.task_id)["mode"] == "audio"

    def test_unavailable_video_is_reported_in_plain_language(self, fake, monkeypatch):
        fake.dummy_ffmpeg()
        monkeypatch.setattr(wk, "check_ytdlp", lambda: "x")
        monkeypatch.setattr(wk, "prepare_cookies_copy", lambda *a, **k: None)

        def unavailable(url, **kw):
            raise u.ContentUnavailableError()

        monkeypatch.setattr(wk, "fetch_metadata", unavailable)
        cfg = tm.create_task("youtube", YT_URL, YT_URL)
        assert run(cfg.task_id) == wk.EXIT_FAILED
        err = ck_of(cfg.task_id)["errors"][-1]
        assert err["code"] == "content_unavailable" and "não está disponível" in err["message"]
