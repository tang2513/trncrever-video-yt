"""Testes de src/task_manager.py: criação, checkpoint, cancelamento, retomada e controle do worker."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from conftest import ROOT, wait_until

import src.task_manager as tm
import src.utils as u
from src.task_manager import CheckpointStore, TaskState
from src.utils import InvalidInputError, WorkerError


def new_task(tmp_path, name="video.mp4", **options):
    src = tmp_path / name
    if not src.exists():
        src.write_bytes(b"conteudo")
    return tm.create_task("local", str(src), src.stem, **options)


def register_worker(task_id, proc, state="transcribing"):
    """Registra um processo existente como 'worker' da tarefa (o ID está na linha de comando dele)."""
    CheckpointStore(task_id, writer="test").update(
        worker={"pid": proc.pid, "create_time": u.get_process_create_time(proc.pid),
                "launched_at": u.now_iso(), "started_at": u.now_iso()},
        state=state,
    )


def set_state(task_id, state, **extra):
    CheckpointStore(task_id, writer="test").update(state=state, **extra)


@pytest.fixture
def dm(data_dir):
    """Atalho: garante a pasta de dados isolada."""
    return data_dir


# ---------------------------------------------------------------------------
# Criação de tarefas
# ---------------------------------------------------------------------------


class TestCreateTask:
    def test_files_and_initial_state(self, dm, tmp_path):
        cfg = new_task(tmp_path, model="base", language="en")
        paths = tm.task_paths(cfg.task_id)
        assert paths.config.is_file() and paths.checkpoint.is_file()
        ck = tm.read_checkpoint(cfg.task_id)
        assert ck["state"] == "pending" and ck["task_id"] == cfg.task_id and ck["model"] == "base"
        assert ck["source_path"] == cfg.source and ck["worker"] is None
        assert tm.load_config(cfg.task_id) == cfg
        assert u.is_valid_task_id(cfg.task_id)

    def test_local_source_skips_subtitles_and_download(self, dm, tmp_path):
        ck = tm.read_checkpoint(new_task(tmp_path).task_id)
        assert ck["steps"]["subtitles"]["status"] == "skipped" and ck["steps"]["download"]["status"] == "skipped"
        assert ck["steps"]["validate"]["status"] == "pending"

    def test_youtube_source_keeps_subtitles_and_download(self, dm):
        cfg = tm.create_task("youtube", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "yt")
        ck = tm.read_checkpoint(cfg.task_id)
        assert ck["steps"]["subtitles"]["status"] == "pending" and ck["steps"]["download"]["status"] == "pending"

    def test_twitch_source_skips_only_subtitles(self, dm):
        ck = tm.read_checkpoint(tm.create_task("twitch", "https://www.twitch.tv/videos/1", "tw").task_id)
        assert ck["steps"]["subtitles"]["status"] == "skipped" and ck["steps"]["download"]["status"] == "pending"

    def test_checkpoint_has_every_field_required_by_the_spec(self, dm, tmp_path):
        ck = tm.read_checkpoint(new_task(tmp_path).task_id)
        for key in ("task_id", "source", "source_path", "model", "language_requested", "device_requested", "state",
                    "total_blocks", "completed_blocks", "current_block", "blocks", "errors", "created_at",
                    "started_at", "finished_at", "temp_paths", "worker", "child_pids", "steps"):
            assert key in ck
        assert set(ck["steps"]) == set(tm.STEP_ORDER) and len(tm.STEP_ORDER) == 9

    @pytest.mark.parametrize("bad", [
        {"model": "gigante"}, {"language": "xx"}, {"device": "tpu"}, {"timestamp_mode": "2min"},
        {"subtitle_mode": "x"}, {"chunk_seconds": 10}, {"chunk_seconds": 99999}, {"beam_size": 0}, {"beam_size": 99},
    ])
    def test_invalid_options_are_rejected_without_creating_anything(self, dm, tmp_path, bad):
        with pytest.raises(InvalidInputError):
            new_task(tmp_path, **bad)
        assert list(tm.TASKS_DIR.iterdir()) == []

    def test_invalid_source_type(self, dm):
        with pytest.raises(InvalidInputError):
            tm.create_task("ftp", "x", "x")

    def test_task_paths_reject_path_traversal(self, dm):
        for bad in ("../x", "..", "abc", ""):
            with pytest.raises(InvalidInputError):
                tm.task_paths(bad)

    def test_same_source_cannot_run_twice(self, dm, tmp_path, spawn_sleeper):
        cfg = new_task(tmp_path)
        register_worker(cfg.task_id, spawn_sleeper(cfg.task_id))
        with pytest.raises(InvalidInputError, match="já está sendo transcrita"):
            new_task(tmp_path)

    def test_config_serialization_is_tolerant_to_unknown_keys(self):
        data = tm.TaskConfig(task_id=u.generate_task_id(), source_type="local", source="x", title="t").to_dict()
        data["campo_futuro"] = 1
        assert tm.TaskConfig.from_dict(data).title == "t"


# ---------------------------------------------------------------------------
# 14. Atualização do checkpoint JSON
# ---------------------------------------------------------------------------


class TestCheckpointStore:
    def test_update_merges_and_records_writer(self, dm, tmp_path):
        tid = new_task(tmp_path).task_id
        store = CheckpointStore(tid, writer="worker")
        before = store.read()["updated_at"]
        time.sleep(1.1)
        ck = store.update(state="transcribing", current_block=3)
        assert ck["state"] == "transcribing" and ck["current_block"] == 3 and ck["last_writer"] == "worker"
        assert ck["updated_at"] != before
        assert tm.read_checkpoint(tid)["current_block"] == 3
        assert ck["model"] == "small"  # o restante foi preservado

    def test_no_temp_or_lock_files_remain(self, dm, tmp_path):
        tid = new_task(tmp_path).task_id
        store = CheckpointStore(tid, writer="worker")
        for i in range(10):
            store.update(n=i)
        names = {p.name for p in tm.task_paths(tid).root.iterdir()}
        assert not any(n.endswith(".tmp") or n.endswith(".lock") for n in names)
        assert {"checkpoint.json", "checkpoint.json.bak", "config.json"} <= names

    def test_corrupted_checkpoint_recovers_from_backup(self, dm, tmp_path):
        tid = new_task(tmp_path).task_id
        store = CheckpointStore(tid, writer="worker")
        store.update(current_block=7)
        tm.task_paths(tid).checkpoint.write_text("{ corrompido", encoding="utf-8")
        assert tm.read_checkpoint(tid)["current_block"] == 7

    def test_lost_checkpoint_is_rebuilt_from_manifest(self, dm, tmp_path):
        tid = new_task(tmp_path).task_id
        paths = tm.task_paths(tid)
        tm.write_manifest(tid, {"mode": "audio", "title": "Palestra", "duration": 1500.0, "chunks": [
            {"index": 0, "start": 0, "end": 600, "duration": 600, "status": "done"},
            {"index": 1, "start": 600, "end": 1200, "duration": 600, "status": "pending"},
        ]})
        paths.checkpoint.unlink()
        u.backup_path(paths.checkpoint).unlink()
        ck = CheckpointStore(tid, writer="worker").read()
        assert ck["state"] == "failed" and ck["completed_blocks"] == [0] and ck["total_blocks"] == 2
        assert ck["title"] == "Palestra" and ck["warnings"]

    def test_concurrent_threads_do_not_lose_updates(self, dm, tmp_path):
        tid = new_task(tmp_path).task_id
        store = CheckpointStore(tid, writer="worker")

        def work():
            for _ in range(20):
                store.mutate(lambda ck: ck.__setitem__("counter", ck.get("counter", 0) + 1))

        threads = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert tm.read_checkpoint(tid)["counter"] == 160

    def test_concurrent_processes_do_not_lose_updates(self, dm, tmp_path):
        """Interface e worker são processos diferentes: o lock de arquivo impede a perda de atualizações."""
        tid = new_task(tmp_path).task_id
        code = textwrap.dedent("""
            import sys
            from src.task_manager import CheckpointStore
            s = CheckpointStore(sys.argv[1], writer="proc")
            for _ in range(25):
                s.mutate(lambda ck: ck.__setitem__("counter", ck.get("counter", 0) + 1))
        """)
        procs = [subprocess.Popen([sys.executable, "-c", code, tid], cwd=str(ROOT)) for _ in range(2)]
        assert all(p.wait(timeout=120) == 0 for p in procs)
        assert tm.read_checkpoint(tid)["counter"] == 50

    def test_throttled_updates_skip_rapid_calls(self, dm, tmp_path):
        tid = new_task(tmp_path).task_id
        store = CheckpointStore(tid, writer="worker")
        assert store.mutate_throttled(lambda ck: ck.__setitem__("a", 1)) is not None
        assert store.mutate_throttled(lambda ck: ck.__setitem__("a", 2)) is None
        assert tm.read_checkpoint(tid)["a"] == 1

    def test_stale_lock_left_by_a_dead_process_does_not_block(self, dm, tmp_path, dead_pid):
        tid = new_task(tmp_path).task_id
        import json

        tm.task_paths(tid).checkpoint_lock.write_text(
            json.dumps({"pid": dead_pid, "create_time": 1.0, "acquired_at": time.time()}))
        CheckpointStore(tid, writer="ui").update(ok=True)
        assert tm.read_checkpoint(tid)["ok"] is True


class TestProgressMath:
    def steps(self, **status):
        base = {s: {"status": "pending", "fraction": 0.0} for s in tm.STEP_ORDER}
        for name, (st, fr) in status.items():
            base[name] = {"status": st, "fraction": fr}
        return {"steps": base}

    def test_weights_sum_to_one(self):
        assert sum(tm.STEP_WEIGHTS.values()) == pytest.approx(1.0)

    def test_skipped_steps_are_excluded(self):
        ck = self.steps(subtitles=("skipped", 0), download=("skipped", 0), validate=("done", 1.0))
        assert tm.compute_overall(ck) == pytest.approx(0.01 / 0.91)

    def test_real_fraction_of_the_current_step_counts(self):
        ck = self.steps(subtitles=("skipped", 0), download=("skipped", 0), validate=("done", 1.0),
                        transcribe=("running", 0.5))
        assert tm.compute_overall(ck) == pytest.approx((0.01 + 0.76 * 0.5) / 0.91)

    def test_never_decreases_as_fraction_grows(self):
        values = []
        for f in (0, 0.1, 0.5, 0.9, 1.0):
            values.append(tm.compute_overall(self.steps(transcribe=("running", f))))
        assert values == sorted(values)

    def test_everything_done_is_100_percent(self):
        ck = {"steps": {s: {"status": "done", "fraction": 1.0} for s in tm.STEP_ORDER}}
        assert tm.compute_overall(ck) == pytest.approx(1.0)
        assert tm.compute_overall({"steps": {}}) == 0.0

    def test_set_step_clamps_fraction(self):
        ck = self.steps()
        tm.set_step(ck, "convert", "running", 7)
        assert ck["steps"]["convert"]["fraction"] == 1.0
        tm.set_step(ck, "convert", fraction=-3)
        assert ck["steps"]["convert"]["fraction"] == 0.0
        tm.set_step(ck, "convert", "done")
        assert ck["steps"]["convert"]["fraction"] == 1.0

    def test_elapsed_time_accumulates_across_runs(self):
        now = 10_000.0
        assert tm.elapsed_seconds({"elapsed_before": 10, "run_started_epoch": now - 5}, now) == pytest.approx(15)
        assert tm.elapsed_seconds({"elapsed_before": 10, "run_started_epoch": None}, now) == 10
        assert tm.elapsed_seconds({}, now) == 0

    def test_remaining_time_uses_measured_speed_only(self):
        now = 1000.0
        ck = {"audio_duration": 1000.0, "transcribed_seconds": 200.0,
              "transcribe_run": {"started_epoch": now - 100, "audio_seconds": 200.0}}
        assert tm.estimate_remaining_seconds(ck, now) == pytest.approx(400.0)  # 2 s de áudio por s real
        ck["transcribe_run"]["audio_seconds"] = 10.0
        assert tm.estimate_remaining_seconds(ck, now) is None  # ainda sem medição confiável
        assert tm.estimate_remaining_seconds({}, now) is None

    def test_messages_and_errors_are_capped(self):
        ck = {}
        for i in range(60):
            tm.add_message(ck, f"m{i}")
        assert len(ck["messages"]) == 40 and ck["message"] == "m59" and ck["messages"][0]["text"] == "m20"
        for i in range(30):
            tm.add_error(ck, u.DownloadError(technical=str(i)))
        assert len(ck["errors"]) == 20


# ---------------------------------------------------------------------------
# 13. Cancelamento
# ---------------------------------------------------------------------------


class TestCancel:
    def test_cancel_file_is_written_and_state_shows_request(self, dm, tmp_path):
        tid = new_task(tmp_path).task_id
        assert not tm.is_cancel_requested(tid)
        tm.request_cancel(tid)
        assert tm.is_cancel_requested(tid) and tm.task_paths(tid).cancel.is_file()
        ck = tm.read_checkpoint(tid)
        assert tm.current_state(ck, tid) == TaskState.CANCEL_REQUESTED
        tm.clear_cancel(tid)
        tm.clear_cancel(tid)  # idempotente
        assert not tm.is_cancel_requested(tid)

    def test_cancel_with_live_worker_leaves_the_finish_to_the_worker(self, dm, tmp_path, spawn_sleeper):
        tid = new_task(tmp_path).task_id
        register_worker(tid, spawn_sleeper(tid))
        tm.request_cancel(tid)
        assert tm.read_checkpoint(tid)["state"] == "transcribing"  # quem muda para 'cancelled' é o worker

    def test_cancel_when_worker_is_already_dead_finishes_immediately(self, dm, tmp_path, dead_pid):
        tid = new_task(tmp_path).task_id
        set_state(tid, "transcribing", worker={"pid": dead_pid, "create_time": 1.0}, completed_blocks=[0, 1])
        tm.request_cancel(tid)
        ck = tm.read_checkpoint(tid)
        assert ck["state"] == "cancelled" and ck["completed_blocks"] == [0, 1] and ck["finished_at"]

    def test_unknown_task(self, dm):
        with pytest.raises(InvalidInputError):
            tm.request_cancel(u.generate_task_id())


# ---------------------------------------------------------------------------
# 17. Processos encerrados inesperadamente
# ---------------------------------------------------------------------------


class TestReconcile:
    def test_dead_worker_marks_task_failed_and_resumable(self, dm, tmp_path, dead_pid):
        tid = new_task(tmp_path).task_id
        set_state(tid, "transcribing", worker={"pid": dead_pid, "create_time": 1.0},
                  run_started_epoch=time.time() - 100, elapsed_before=5.0)
        ck = tm.reconcile_task(tid)
        assert ck["state"] == "failed" and ck["finished_at"]
        assert ck["errors"][-1]["code"] == "worker" and "retomar" in ck["errors"][-1]["message"]
        assert ck["run_started_epoch"] is None and ck["elapsed_before"] == pytest.approx(105, abs=3)
        assert tm.summarize(ck).resumable and not tm.summarize(ck).alive

    def test_cancel_requested_and_dead_worker_becomes_cancelled(self, dm, tmp_path, dead_pid):
        tid = new_task(tmp_path).task_id
        set_state(tid, "cancel_requested", worker={"pid": dead_pid, "create_time": 1.0})
        u.atomic_write_json(tm.task_paths(tid).cancel, {"requested_at": u.now_iso()}, keep_backup=False)
        assert tm.reconcile_task(tid)["state"] == "cancelled"

    def test_live_worker_is_left_alone(self, dm, tmp_path, spawn_sleeper):
        tid = new_task(tmp_path).task_id
        register_worker(tid, spawn_sleeper(tid))
        assert tm.reconcile_task(tid)["state"] == "transcribing" and tm.is_worker_alive(tid)

    def test_terminal_states_are_never_changed(self, dm, tmp_path, dead_pid):
        tid = new_task(tmp_path).task_id
        set_state(tid, "completed", worker={"pid": dead_pid, "create_time": 1.0})
        assert tm.reconcile_task(tid)["state"] == "completed"

    def test_pending_task_that_was_never_launched_is_left_alone(self, dm, tmp_path):
        assert tm.reconcile_task(new_task(tmp_path).task_id)["state"] == "pending"

    def test_reused_pid_is_not_mistaken_for_the_worker(self, dm, tmp_path, spawn_sleeper):
        """O PID existe (outro programa o reaproveitou), mas o horário de criação não confere."""
        tid = new_task(tmp_path).task_id
        p = spawn_sleeper(tid)
        set_state(tid, "transcribing", worker={"pid": p.pid, "create_time": u.get_process_create_time(p.pid) - 3600})
        assert not tm.is_worker_alive(tid)
        assert tm.reconcile_task(tid)["state"] == "failed"

    def test_process_without_the_task_id_in_its_command_line_is_not_the_worker(self, dm, tmp_path, spawn_sleeper):
        tid = new_task(tmp_path).task_id
        p = spawn_sleeper("outro-programa")
        register_worker(tid, p)
        assert not tm.is_worker_alive(tid)

    def test_orphan_ffmpeg_is_killed(self, dm, tmp_path, spawn_sleeper, dead_pid):
        tid = new_task(tmp_path).task_id
        child = spawn_sleeper("ffmpeg-simulado")
        set_state(tid, "extracting_audio", worker={"pid": dead_pid, "create_time": 1.0},
                  child_pids=[{"pid": child.pid, "create_time": u.get_process_create_time(child.pid)}])
        ck = tm.reconcile_task(tid)
        assert wait_until(lambda: child.poll() is not None, 10)
        assert ck["child_pids"] == []

    def test_missing_task(self, dm):
        assert tm.reconcile_task(u.generate_task_id()) is None


# ---------------------------------------------------------------------------
# 12. Retomada
# ---------------------------------------------------------------------------


class TestResume:
    def test_prepare_resume_resets_state_and_keeps_progress(self, dm, tmp_path, dead_pid):
        tid = new_task(tmp_path).task_id
        set_state(tid, "failed", worker={"pid": dead_pid, "create_time": 1.0}, completed_blocks=[0, 1, 2],
                  total_blocks=8, current_block=3, finished_at=u.now_iso())
        tm.request_cancel(tid)  # um pedido de cancelamento antigo não pode impedir a retomada
        tm.prepare_resume(tid)
        ck = tm.read_checkpoint(tid)
        assert ck["state"] == "pending" and ck["worker"] is None and ck["resume_count"] == 1
        assert ck["completed_blocks"] == [0, 1, 2] and ck["current_block"] is None and ck["finished_at"] is None
        assert not tm.is_cancel_requested(tid)

    def test_resume_on_cpu_updates_the_config(self, dm, tmp_path):
        tid = new_task(tmp_path, device="cuda", allow_cpu_fallback=False).task_id
        set_state(tid, "failed")
        cfg = tm.prepare_resume(tid, device_override="cpu")
        assert cfg.device == "cpu" and cfg.allow_cpu_fallback
        assert tm.load_config(tid).device == "cpu"
        assert tm.read_checkpoint(tid)["device_requested"] == "cpu"

    def test_completed_task_cannot_be_resumed(self, dm, tmp_path):
        tid = new_task(tmp_path).task_id
        set_state(tid, "completed")
        with pytest.raises(WorkerError, match="concluída"):
            tm.prepare_resume(tid)

    def test_running_task_cannot_be_resumed(self, dm, tmp_path, spawn_sleeper):
        tid = new_task(tmp_path).task_id
        register_worker(tid, spawn_sleeper(tid))
        with pytest.raises(WorkerError, match="em execução"):
            tm.prepare_resume(tid)

    def test_orphans_are_cleaned_on_resume(self, dm, tmp_path, spawn_sleeper, dead_pid):
        tid = new_task(tmp_path).task_id
        child = spawn_sleeper("ffmpeg-simulado")
        set_state(tid, "failed", worker={"pid": dead_pid, "create_time": 1.0},
                  child_pids=[{"pid": child.pid, "create_time": u.get_process_create_time(child.pid)}])
        tm.prepare_resume(tid)
        assert wait_until(lambda: child.poll() is not None, 10)


# ---------------------------------------------------------------------------
# 15–16. Inicialização do worker e comunicação worker ↔ interface (processo real)
# ---------------------------------------------------------------------------


class TestStartWorker:
    def test_real_worker_process_reports_failure_through_the_checkpoint(self, dm, tmp_path):
        cfg = tm.create_task("local", str(tmp_path / "nao_existe.mp4"), "nao_existe")
        pid = tm.start_worker(cfg.task_id)
        assert pid > 0
        assert wait_until(lambda: tm.TaskState((tm.read_checkpoint(cfg.task_id) or {}).get("state", "pending")).is_terminal, 120)
        ck = tm.read_checkpoint(cfg.task_id)
        assert ck["state"] == "failed"
        assert ck["errors"][-1]["code"] in ("invalid_input", "ffmpeg_not_found")  # mensagem amigável, não traceback
        assert ck["worker"]["pid"] and ck["worker"]["started_at"] and ck["finished_at"]
        assert wait_until(lambda: not u.is_process_alive(ck["worker"]["pid"]), 15)
        assert not tm.task_paths(cfg.task_id).worker_lock.exists() and ck["child_pids"] == []
        log = tm.read_log_tail(cfg.task_id)
        assert "Iniciando worker" in tm.task_paths(cfg.task_id).log_file.read_text(encoding="utf-8")
        assert log is not None

    def test_worker_that_dies_on_startup_is_reported(self, dm, tmp_path):
        cfg = new_task(tmp_path)
        paths = tm.task_paths(cfg.task_id)
        paths.config.write_text("{ corrompido", encoding="utf-8")
        u.backup_path(paths.config).write_text("{ corrompido", encoding="utf-8")
        with pytest.raises(WorkerError, match="não conseguiu iniciar"):
            tm.start_worker(cfg.task_id)
        assert tm.read_checkpoint(cfg.task_id)["state"] == "failed"

    def test_only_one_worker_at_a_time(self, dm, tmp_path, spawn_sleeper):
        a = new_task(tmp_path, "a.mp4")
        register_worker(a.task_id, spawn_sleeper(a.task_id))
        b = new_task(tmp_path, "b.mp4")
        with pytest.raises(WorkerError, match="em andamento"):
            tm.start_worker(b.task_id)

    def test_same_task_cannot_be_started_twice(self, dm, tmp_path, spawn_sleeper):
        a = new_task(tmp_path)
        register_worker(a.task_id, spawn_sleeper(a.task_id))
        with pytest.raises(WorkerError, match="já está em execução"):
            tm.start_worker(a.task_id)

    def test_completed_and_missing_tasks(self, dm, tmp_path):
        a = new_task(tmp_path)
        set_state(a.task_id, "completed")
        with pytest.raises(WorkerError, match="concluída"):
            tm.start_worker(a.task_id)
        with pytest.raises(WorkerError):
            tm.start_worker(u.generate_task_id())

    def test_active_task_ids(self, dm, tmp_path, spawn_sleeper):
        a = new_task(tmp_path)
        assert tm.active_task_ids() == []
        register_worker(a.task_id, spawn_sleeper(a.task_id))
        assert tm.active_task_ids() == [a.task_id]


# ---------------------------------------------------------------------------
# Listagem, exclusão e manifesto
# ---------------------------------------------------------------------------


class TestListing:
    def test_newest_first_and_summary(self, dm, tmp_path):
        first = new_task(tmp_path, "um.mp4").task_id
        time.sleep(1.05)  # IDs começam com data/hora (resolução de 1 s)
        second = new_task(tmp_path, "dois.mp4").task_id
        set_state(first, "failed", completed_blocks=[0], total_blocks=4)
        tasks = tm.list_tasks()
        assert [t.task_id for t in tasks] == [second, first]
        t = {x.task_id: x for x in tasks}[first]
        assert t.state == TaskState.FAILED and t.resumable and (t.completed_blocks, t.total_blocks) == (1, 4)
        assert [x.task_id for x in tm.find_interrupted_tasks()] == [first]

    def test_delete_only_removes_task_files(self, dm, tmp_path):
        cfg = new_task(tmp_path, "original.mp4")
        original = Path(cfg.source)
        tm.task_paths(cfg.task_id).log_file.write_text("log", encoding="utf-8")
        results = dm / "transcricoes" / "resultado"
        results.mkdir(parents=True)
        (results / "resultado.txt").write_text("texto", encoding="utf-8")
        tm.delete_task(cfg.task_id)
        assert not tm.task_paths(cfg.task_id).root.exists() and not tm.task_paths(cfg.task_id).log_file.exists()
        assert original.is_file() and (results / "resultado.txt").is_file()

    def test_cannot_delete_running_task(self, dm, tmp_path, spawn_sleeper):
        cfg = new_task(tmp_path)
        register_worker(cfg.task_id, spawn_sleeper(cfg.task_id))
        with pytest.raises(WorkerError):
            tm.delete_task(cfg.task_id)

    def test_manifest_roundtrip(self, dm, tmp_path):
        tid = new_task(tmp_path).task_id
        assert tm.read_manifest(tid) == {}
        tm.write_manifest(tid, {"mode": "audio", "chunks": [{"index": 0}]})
        m = tm.read_manifest(tid)
        assert m["mode"] == "audio" and m["schema"] == tm.SCHEMA_VERSION and m["updated_at"]
