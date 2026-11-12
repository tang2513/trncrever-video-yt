"""Testes de src/utils.py: caminhos, formatos, timestamps, JSON atômico, locks, processos e erros."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import wait_until

import src.utils as u


# ---------------------------------------------------------------------------
# 1. Validação de caminhos locais
# ---------------------------------------------------------------------------


class TestValidateLocalFile:
    def test_valid_file(self, tmp_path):
        f = tmp_path / "gravacao.mp3"
        f.write_bytes(b"abc")
        assert u.validate_local_file(str(f)) == f

    def test_quotes_and_spaces_are_removed(self, tmp_path):
        f = tmp_path / "minha gravação.wav"
        f.write_bytes(b"abc")
        assert u.validate_local_file(f'  "{f}"  ') == f

    def test_empty_path(self):
        with pytest.raises(u.InvalidInputError):
            u.validate_local_file("   ")

    def test_missing_file(self, tmp_path):
        with pytest.raises(u.InvalidInputError, match="não encontrado"):
            u.validate_local_file(str(tmp_path / "nao_existe.mp4"))

    def test_directory(self, tmp_path):
        with pytest.raises(u.InvalidInputError, match="pasta"):
            u.validate_local_file(str(tmp_path))

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "texto.txt"
        f.write_text("x")
        with pytest.raises(u.UnsupportedFormatError):
            u.validate_local_file(str(f))

    def test_empty_file_is_reported_as_corrupted(self, tmp_path):
        f = tmp_path / "vazio.mp4"
        f.write_bytes(b"")
        with pytest.raises(u.CorruptedMediaError):
            u.validate_local_file(str(f))

    def test_never_modifies_the_file(self, tmp_path):
        f = tmp_path / "a.mp4"
        f.write_bytes(b"conteudo")
        before = (f.read_bytes(), f.stat().st_mtime_ns)
        u.validate_local_file(str(f))
        assert (f.read_bytes(), f.stat().st_mtime_ns) == before


# ---------------------------------------------------------------------------
# 2. Identificação de formatos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.mp4", "video"), ("a.MKV", "video"), ("a.mov", "video"), ("a.avi", "video"), ("a.webm", "video"),
        ("a.mp3", "audio"), ("a.WAV", "audio"), ("a.m4a", "audio"), ("a.flac", "audio"),
        ("a.txt", None), ("a", None), ("a.mp4.exe", None),
    ],
)
def test_classify_media_extension(name, expected):
    assert u.classify_media_extension(name) == expected


def test_supported_extensions_match_the_spec():
    assert u.SUPPORTED_EXTENSIONS == {".mp4", ".mkv", ".mov", ".avi", ".webm", ".mp3", ".wav", ".m4a", ".flac"}


# ---------------------------------------------------------------------------
# 7. Conversão de timestamps
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_formats(self):
        assert u.format_timestamp(3661.5, "txt") == "01:01:01"
        assert u.format_timestamp(3661.5, "srt") == "01:01:01,500"
        assert u.format_timestamp(3661.5, "vtt") == "01:01:01.500"

    def test_txt_truncates_never_rounds_up(self):
        assert u.format_timestamp(59.9, "txt") == "00:00:59"

    def test_negative_and_nan(self):
        assert u.format_timestamp(-5) == "00:00:00"
        assert u.format_timestamp(float("nan")) == "00:00:00"

    def test_more_than_99_hours(self):
        assert u.format_timestamp(100 * 3600, "txt") == "100:00:00"

    def test_unknown_style(self):
        with pytest.raises(ValueError):
            u.format_timestamp(1, "xyz")

    @pytest.mark.parametrize(
        "text,seconds",
        [("01:02:03.500", 3723.5), ("02:03,250", 123.25), ("1:2:3", 3723.0), ("00:00:00.5", 0.5), ("00:00:01.050", 1.05)],
    )
    def test_parse(self, text, seconds):
        assert u.parse_timestamp(text) == pytest.approx(seconds)

    @pytest.mark.parametrize("bad", ["", "abc", "12", "1:2:3:4", "aa:bb:cc"])
    def test_parse_invalid(self, bad):
        with pytest.raises(ValueError):
            u.parse_timestamp(bad)

    @pytest.mark.parametrize("seconds", [0, 0.001, 59.999, 3600, 7322.123, 14399.5])
    def test_roundtrip(self, seconds):
        assert u.parse_timestamp(u.format_timestamp(seconds, "vtt")) == pytest.approx(seconds, abs=0.001)

    def test_duration_and_bytes(self):
        assert u.format_duration(8) == "8s"
        assert u.format_duration(760) == "12min 40s"
        assert u.format_duration(7503) == "2h 05min 03s"
        assert u.format_bytes(0) == "0 B"
        assert u.format_bytes(1536) == "1.5 KB"
        assert u.format_bytes(10 * 1024 ** 3) == "10.0 GB"


# ---------------------------------------------------------------------------
# Nomes e IDs
# ---------------------------------------------------------------------------


class TestNamesAndIds:
    def test_sanitize_removes_invalid_characters(self):
        out = u.sanitize_filename('vídeo: "teste"/final?*<>|')
        assert not set('<>:"/\\|?*') & set(out)
        assert out.startswith("vídeo")

    def test_sanitize_reserved_and_empty(self):
        assert u.sanitize_filename("CON") == "_CON"
        assert u.sanitize_filename("   ...  ") == "transcricao"
        assert u.sanitize_filename("") == "transcricao"

    def test_sanitize_truncates(self):
        assert len(u.sanitize_filename("a" * 500, max_length=40)) <= 40

    def test_unique_path_never_overwrites(self, tmp_path):
        base = tmp_path / "x.txt"
        base.write_text("1")
        second = u.unique_path(base)
        assert second.name == "x (2).txt"
        second.write_text("2")
        assert u.unique_path(base).name == "x (3).txt"

    def test_unique_output_dir(self, tmp_path):
        (tmp_path / "Meu Video").mkdir()
        assert u.unique_output_dir(tmp_path, "Meu Video").name == "Meu Video (2)"

    def test_task_ids(self):
        tid = u.generate_task_id()
        assert u.is_valid_task_id(tid)
        assert tid != u.generate_task_id()
        for bad in ("", "../x", "20240101-000000-ZZZZZZZZ", "abc", "20240101-000000-abcdef12/.."):
            assert not u.is_valid_task_id(bad)

    def test_is_relative_to(self, tmp_path):
        assert u.is_relative_to(tmp_path / "a" / "b", tmp_path)
        assert not u.is_relative_to(tmp_path.parent, tmp_path)


# ---------------------------------------------------------------------------
# 14. JSON atômico
# ---------------------------------------------------------------------------


class TestAtomicJson:
    def test_roundtrip_utf8(self, tmp_path):
        p = tmp_path / "d.json"
        data = {"título": "Transcrição ção", "n": 3, "p": tmp_path}
        u.atomic_write_json(p, data)
        loaded = u.read_json(p)
        assert loaded["título"] == "Transcrição ção"
        assert loaded["p"] == str(tmp_path)
        assert "ç" in p.read_text(encoding="utf-8")  # ensure_ascii=False

    def test_no_temp_files_left(self, tmp_path):
        p = tmp_path / "d.json"
        for i in range(5):
            u.atomic_write_json(p, {"i": i})
        leftovers = [f.name for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
        assert leftovers == []

    def test_falls_back_to_backup_when_main_is_corrupted(self, tmp_path):
        p = tmp_path / "d.json"
        u.atomic_write_json(p, {"ok": True})
        p.write_text("{ isso nao e json", encoding="utf-8")
        assert u.read_json(p) == {"ok": True}

    def test_both_corrupted(self, tmp_path):
        p = tmp_path / "d.json"
        u.atomic_write_json(p, {"ok": True})
        p.write_text("{", encoding="utf-8")
        u.backup_path(p).write_text("{", encoding="utf-8")
        assert u.read_json(p, default=None) is None
        with pytest.raises(u.CorruptedJSONError):
            u.read_json(p)

    def test_missing_file(self, tmp_path):
        assert u.read_json(tmp_path / "x.json", default={"d": 1}) == {"d": 1}
        with pytest.raises(FileNotFoundError):
            u.read_json(tmp_path / "x.json")

    def test_previous_state_survives_a_failed_replace(self, tmp_path, monkeypatch):
        p = tmp_path / "d.json"
        u.atomic_write_json(p, {"v": 1})

        def boom(*a, **k):
            raise u.CheckpointError(technical="disco cheio simulado")

        monkeypatch.setattr(u, "_replace_with_retry", boom)
        with pytest.raises(u.CheckpointError):
            u.atomic_write_json(p, {"v": 2})
        monkeypatch.undo()
        assert u.read_json(p) == {"v": 1}
        assert [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")] == []

    def test_concurrent_writers_never_produce_invalid_json(self, tmp_path):
        p = tmp_path / "c.json"
        u.atomic_write_json(p, {"n": 0})
        stop = threading.Event()
        errors: list[str] = []

        def reader():
            while not stop.is_set():
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        json.load(fh)
                except (PermissionError, FileNotFoundError):
                    pass  # Windows: arquivo em substituição naquele instante
                except Exception as exc:  # noqa: BLE001
                    errors.append(repr(exc))

        t = threading.Thread(target=reader)
        t.start()
        for i in range(60):
            u.atomic_write_json(p, {"n": i, "texto": "x" * 2000}, keep_backup=False)
        stop.set()
        t.join()
        assert errors == []


# ---------------------------------------------------------------------------
# Lock entre processos
# ---------------------------------------------------------------------------


class TestFileLock:
    def test_acquire_release(self, tmp_path):
        lock = u.FileLock(tmp_path / "a.lock")
        with lock:
            assert (tmp_path / "a.lock").exists()
        assert not (tmp_path / "a.lock").exists()

    def test_second_holder_times_out(self, tmp_path):
        with u.FileLock(tmp_path / "a.lock"):
            with pytest.raises(u.CheckpointError):
                u.FileLock(tmp_path / "a.lock", timeout=0.3, stale_after=10 ** 9).acquire()

    def test_stale_lock_from_dead_process_is_removed(self, tmp_path, dead_pid):
        lock_path = tmp_path / "a.lock"
        lock_path.write_text(json.dumps({"pid": dead_pid, "create_time": 1.0, "acquired_at": time.time()}))
        with u.FileLock(lock_path, timeout=2):
            pass

    def test_lock_of_live_process_is_respected(self, tmp_path):
        ident = u.current_process_identity()
        (tmp_path / "a.lock").write_text(json.dumps({**ident, "acquired_at": time.time()}))
        with pytest.raises(u.CheckpointError):
            u.FileLock(tmp_path / "a.lock", timeout=0.3, stale_after=10 ** 9).acquire()

    def test_mutual_exclusion_between_threads(self, tmp_path):
        counter = {"v": 0}
        path = tmp_path / "c.lock"

        def work():
            for _ in range(15):
                with u.FileLock(path, timeout=30):
                    v = counter["v"]
                    time.sleep(0.001)
                    counter["v"] = v + 1

        threads = [threading.Thread(target=work) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert counter["v"] == 60


# ---------------------------------------------------------------------------
# 17. Identificação de processos
# ---------------------------------------------------------------------------


class TestProcessIdentity:
    def test_current_process(self):
        ident = u.current_process_identity()
        assert u.is_process_alive(ident["pid"], ident["create_time"])

    def test_wrong_create_time_means_pid_was_reused(self):
        ident = u.current_process_identity()
        assert not u.is_process_alive(ident["pid"], ident["create_time"] - 3600)

    def test_none_and_dead(self, dead_pid):
        assert not u.is_process_alive(None)
        assert not u.is_process_alive(0)
        assert not u.is_process_alive(dead_pid)

    def test_cmdline_marker(self, spawn_sleeper):
        p = spawn_sleeper("marcador-unico-123")
        ct = u.get_process_create_time(p.pid)
        assert u.is_process_alive(p.pid, ct, cmdline_marker="marcador-unico-123")
        assert not u.is_process_alive(p.pid, ct, cmdline_marker="outro-marcador")

    def test_kill_process_tree_kills_children(self, tmp_path):
        child_pid_file = tmp_path / "child.pid"
        code = (
            "import subprocess, sys, time\n"
            "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            f"open(r'{child_pid_file}', 'w').write(str(c.pid))\n"
            "time.sleep(120)\n"
        )
        parent = subprocess.Popen([sys.executable, "-c", code])
        try:
            assert wait_until(child_pid_file.exists, 20)
            time.sleep(0.2)
            child_pid = int(child_pid_file.read_text())
            killed = u.kill_process_tree(parent.pid)
            assert parent.pid in killed
            assert wait_until(lambda: not u.is_process_alive(child_pid), 10)
            assert wait_until(lambda: parent.poll() is not None, 10)
        finally:
            if parent.poll() is None:
                parent.kill()

    def test_kill_nonexistent_is_harmless(self, dead_pid):
        assert u.kill_process_tree(dead_pid) == []


# ---------------------------------------------------------------------------
# Execução de processos externos
# ---------------------------------------------------------------------------


class TestRunProcess:
    def test_captures_output_and_callbacks(self):
        out, err = [], []
        r = u.run_process(
            [sys.executable, "-c", "import sys; print('olá'); print('erro', file=sys.stderr)"],
            on_stdout_line=out.append, on_stderr_line=err.append,
        )
        assert r.returncode == 0
        assert out == ["olá"] and err == ["erro"]
        assert r.stdout_tail == ["olá"]

    def test_nonzero_exit_code(self):
        r = u.run_process([sys.executable, "-c", "import sys; sys.exit(7)"])
        assert r.returncode == 7

    def test_large_output_does_not_deadlock(self):
        r = u.run_process([sys.executable, "-c", "print('x' * 100 + '\\n') or [print('y' * 200) for _ in range(20000)]"],
                          timeout=60, tail_lines=5)
        assert r.returncode == 0
        assert len(r.stdout_tail) == 5

    def test_cancel_kills_the_process(self):
        started = []
        t0 = time.monotonic()
        with pytest.raises(u.TaskCancelledError):
            u.run_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cancel_check=lambda: time.monotonic() - t0 > 0.6,
                on_start=started.append,
            )
        assert time.monotonic() - t0 < 15
        assert wait_until(lambda: not u.is_process_alive(started[0]), 10)

    def test_on_start_and_on_exit(self):
        events = []
        u.run_process([sys.executable, "-c", "pass"], on_start=lambda p: events.append(("s", p)),
                      on_exit=lambda p: events.append(("e", p)))
        assert [e[0] for e in events] == ["s", "e"] and events[0][1] == events[1][1]

    def test_timeout(self):
        with pytest.raises(u.ProcessTimeoutError):
            u.run_process([sys.executable, "-c", "import time; time.sleep(60)"], timeout=1)

    def test_idle_timeout(self):
        with pytest.raises(u.ProcessTimeoutError):
            u.run_process([sys.executable, "-c", "import time; time.sleep(60)"], idle_timeout=1)

    def test_missing_program(self):
        with pytest.raises(u.TranscriberError, match="não encontrado"):
            u.run_process(["programa_que_nao_existe_xyz"])


# ---------------------------------------------------------------------------
# Novas tentativas
# ---------------------------------------------------------------------------


class TestRetry:
    def test_recovers_after_temporary_failures(self):
        calls = {"n": 0}
        seen = []

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise u.NetworkError(technical="x")
            return "ok"

        assert u.retry_call(flaky, base_delay=0.01, on_retry=lambda a, d, e: seen.append((a, d))) == "ok"
        assert calls["n"] == 3
        assert seen == [(1, 0.01), (2, 0.02)]  # intervalos progressivos

    def test_gives_up_after_limit(self):
        calls = {"n": 0}

        def always():
            calls["n"] += 1
            raise u.NetworkError(technical="x")

        with pytest.raises(u.NetworkError):
            u.retry_call(always, attempts=3, base_delay=0.01)
        assert calls["n"] == 3

    def test_non_retryable_error_is_not_retried(self):
        calls = {"n": 0}

        def bad():
            calls["n"] += 1
            raise u.ContentUnavailableError()

        with pytest.raises(u.ContentUnavailableError):
            u.retry_call(bad, base_delay=0.01)
        assert calls["n"] == 1

    def test_cancel_during_wait(self):
        def always():
            raise u.NetworkError(technical="x")

        with pytest.raises(u.TaskCancelledError):
            u.retry_call(always, base_delay=30, cancel_check=lambda: True)

    def test_delay_is_capped(self):
        delays = []

        def always():
            raise u.NetworkError(technical="x")

        with pytest.raises(u.NetworkError):
            u.retry_call(always, attempts=6, base_delay=0.001, max_delay=0.002, on_retry=lambda a, d, e: delays.append(d))
        assert max(delays) == 0.002


# ---------------------------------------------------------------------------
# 18. Erros e disco
# ---------------------------------------------------------------------------


class TestErrors:
    def test_friendly_messages(self):
        assert u.friendly_error(MemoryError()) == u.InsufficientMemoryError.default_message
        assert "permissão" in u.friendly_error(PermissionError()).lower() or "negou" in u.friendly_error(PermissionError())
        assert "não foi encontrado" in u.friendly_error(FileNotFoundError())
        assert u.friendly_error(u.DiskSpaceError()) == u.DiskSpaceError.default_message
        assert "ValueError" in u.friendly_error(ValueError("x"))

    def test_disk_full_oserror(self):
        import errno

        assert u.friendly_error(OSError(errno.ENOSPC, "no space")) == u.DiskSpaceError.default_message

    def test_error_to_dict(self):
        d = u.error_to_dict(u.DownloadError(technical="detalhe"))
        assert d["code"] == "download" and d["technical"] == "detalhe" and d["time"]
        d2 = u.error_to_dict(RuntimeError("boom"))
        assert d2["code"] == "unexpected" and "boom" in d2["technical"]

    def test_all_errors_have_user_message(self):
        for cls in [c for c in vars(u).values() if isinstance(c, type) and issubclass(c, u.TranscriberError)]:
            assert cls().user_message

    def test_ensure_disk_space(self, tmp_path):
        u.ensure_disk_space(tmp_path, 1024)
        with pytest.raises(u.DiskSpaceError):
            u.ensure_disk_space(tmp_path, 1 << 62)
        # caminho que ainda não existe usa o volume do ancestral existente
        assert u.get_free_disk_space(tmp_path / "a" / "b") > 0

    def test_wav_size_estimate(self):
        assert u.estimate_wav_bytes(1.0) == 32000 + 4096
        assert u.estimate_wav_bytes(-5) == 4096


def test_task_state_helpers():
    assert u.TaskState.COMPLETED.is_terminal and not u.TaskState.TRANSCRIBING.is_terminal
    assert u.TaskState.FAILED in u.RESUMABLE_STATES and u.TaskState.CANCELLED in u.RESUMABLE_STATES
    assert u.TaskState.COMPLETED not in u.RESUMABLE_STATES
    assert {s.value for s in u.TaskState} == {
        "pending", "validating", "downloading", "extracting_audio", "splitting", "loading_model",
        "transcribing", "finalizing", "completed", "cancel_requested", "cancelled", "failed",
    }
    assert all(s.label for s in u.TaskState)
