"""Testes de src/file_picker.py (o diálogo em si não é aberto; o subprocesso é simulado)."""

from __future__ import annotations

import base64
import subprocess

import pytest

import src.file_picker as fp


def test_filter_lists_all_supported_formats():
    flt = fp.build_filter()
    for ext in ("mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a", "flac"):
        assert f"*.{ext}" in flt
    assert flt.endswith("Todos os arquivos (*.*)|*.*")


def test_script_escapes_single_quotes():
    script = fp.build_powershell_script("Título do O'Brien", r"C:\Users\O'Neil\Vídeos")
    assert "O''Brien" in script and "O''Neil" in script
    assert "ShowDialog" in script and "OpenFileDialog" in script
    assert "InitialDirectory" in script


def test_script_without_initial_dir():
    assert "InitialDirectory" not in fp.build_powershell_script("t")


def test_encoded_command_roundtrip_preserves_accents():
    script = "Write-Output 'seleção de vídeo'"
    decoded = base64.b64decode(fp.encode_command(script)).decode("utf-16-le")
    assert decoded == script


@pytest.mark.parametrize("raw,expected", [
    (b"\xef\xbb\xbfC:\\Videos\\a\xc3\xa7\xc3\xa3o.mp4\r\n", "C:\\Videos\\ação.mp4"),
    (b"C:\\x.mp4", "C:\\x.mp4"),
    (b"", None),
    (b"\r\n  \r\n", None),
])
def test_parse_output(raw, expected):
    assert fp.parse_picker_output(raw) == expected


class TestPickFile:
    @pytest.fixture(autouse=True)
    def _windows_with_powershell(self, monkeypatch):
        monkeypatch.setattr(fp, "IS_WINDOWS", True)
        monkeypatch.setattr(fp, "_find_powershell", lambda: "powershell.exe")

    def test_returns_chosen_path(self, monkeypatch):
        captured = {}

        def fake_run(args, **kw):
            captured["args"] = args
            return subprocess.CompletedProcess(args, 0, stdout=b"\xef\xbb\xbfC:\\v\\a.mp4", stderr=b"")

        monkeypatch.setattr(fp.subprocess, "run", fake_run)
        assert fp.pick_file() == "C:\\v\\a.mp4"
        args = captured["args"]
        assert "-STA" in args and "-EncodedCommand" in args and "-NoProfile" in args

    def test_user_cancelled_returns_none(self, monkeypatch):
        monkeypatch.setattr(fp.subprocess, "run",
                            lambda args, **kw: subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b""))
        assert fp.pick_file() is None

    def test_powershell_failure(self, monkeypatch):
        monkeypatch.setattr(fp.subprocess, "run",
                            lambda args, **kw: subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"erro"))
        with pytest.raises(fp.FilePickerError) as exc:
            fp.pick_file()
        assert "caminho" in exc.value.user_message

    def test_timeout(self, monkeypatch):
        def fake_run(args, **kw):
            raise subprocess.TimeoutExpired(args, 1)

        monkeypatch.setattr(fp.subprocess, "run", fake_run)
        with pytest.raises(fp.FilePickerError):
            fp.pick_file(timeout=1)

    def test_cannot_start(self, monkeypatch):
        def fake_run(args, **kw):
            raise OSError("sem permissão")

        monkeypatch.setattr(fp.subprocess, "run", fake_run)
        with pytest.raises(fp.FilePickerError):
            fp.pick_file()

    def test_initial_dir_only_used_when_it_exists(self, monkeypatch, tmp_path):
        scripts = []

        def fake_run(args, **kw):
            scripts.append(base64.b64decode(args[args.index("-EncodedCommand") + 1]).decode("utf-16-le"))
            return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

        monkeypatch.setattr(fp.subprocess, "run", fake_run)
        fp.pick_file(initial_dir=tmp_path)
        fp.pick_file(initial_dir=tmp_path / "nao_existe")
        assert "InitialDirectory" in scripts[0] and "InitialDirectory" not in scripts[1]


def test_not_windows(monkeypatch):
    monkeypatch.setattr(fp, "IS_WINDOWS", False)
    with pytest.raises(fp.FilePickerError):
        fp.pick_file()


def test_no_powershell(monkeypatch):
    monkeypatch.setattr(fp, "IS_WINDOWS", True)
    monkeypatch.setattr(fp, "_find_powershell", lambda: None)
    with pytest.raises(fp.FilePickerError):
        fp.pick_file()
