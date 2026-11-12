"""Seletor nativo de arquivos do Windows, executado em processo isolado.

Não usa ``tkinter`` na thread do Streamlit. Em vez disso, abre um processo
PowerShell em modo STA (exigido pelo Windows Forms) que exibe o
``System.Windows.Forms.OpenFileDialog`` e imprime o caminho escolhido.

* Cancelamento do usuário → retorna ``None``.
* Falha ao iniciar o PowerShell/diálogo → ``FilePickerError`` (a interface
  continua permitindo digitar o caminho manualmente).
* O script segue em ``-EncodedCommand`` (UTF-16LE em Base64), o que preserva
  acentos no título e nos caminhos sem problemas de aspas/codificação.
* O arquivo escolhido nunca é aberto ou alterado aqui.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .utils import CREATE_NO_WINDOW, IS_WINDOWS, SUPPORTED_EXTENSIONS, TranscriberError

logger = logging.getLogger(__name__)


class FilePickerError(TranscriberError):
    code = "file_picker"
    default_message = (
        "Não foi possível abrir o seletor de arquivos do Windows. "
        "Digite ou cole o caminho completo do arquivo no campo de texto."
    )


def _ps_quote(value: str) -> str:
    """Literal de string do PowerShell entre aspas simples (dobra as aspas internas)."""
    return "'" + value.replace("'", "''") + "'"


def build_filter() -> str:
    exts = sorted(SUPPORTED_EXTENSIONS)
    patterns = ";".join(f"*{e}" for e in exts)
    return f"Vídeo e áudio ({patterns})|{patterns}|Todos os arquivos (*.*)|*.*"


def build_powershell_script(title: str, initial_dir: Optional[str] = None, file_filter: Optional[str] = None) -> str:
    """Script PowerShell que mostra o diálogo e escreve o caminho em stdout (UTF-8)."""
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8",
        "Add-Type -AssemblyName System.Windows.Forms",
        "Add-Type -AssemblyName System.Drawing",
        "[System.Windows.Forms.Application]::EnableVisualStyles()",
        # Janela "dona" invisível e sempre no topo: faz o diálogo aparecer na frente do navegador.
        "$owner = New-Object System.Windows.Forms.Form",
        "$owner.TopMost = $true",
        "$owner.ShowInTaskbar = $false",
        "$owner.StartPosition = 'Manual'",
        "$owner.Location = New-Object System.Drawing.Point(-32000, -32000)",
        "$owner.Size = New-Object System.Drawing.Size(1, 1)",
        "$owner.Show()",
        "$dialog = New-Object System.Windows.Forms.OpenFileDialog",
        f"$dialog.Title = {_ps_quote(title)}",
        f"$dialog.Filter = {_ps_quote(file_filter or build_filter())}",
        "$dialog.CheckFileExists = $true",
        "$dialog.CheckPathExists = $true",
        "$dialog.Multiselect = $false",
        "$dialog.RestoreDirectory = $true",
    ]
    if initial_dir:
        lines.append(f"$dialog.InitialDirectory = {_ps_quote(initial_dir)}")
    lines += [
        "$result = $dialog.ShowDialog($owner)",
        "$owner.Close()",
        "if ($result -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($dialog.FileName) }",
    ]
    return "\n".join(lines)


def encode_command(script: str) -> str:
    """Formato exigido por ``powershell -EncodedCommand``: Base64 de UTF-16LE."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _find_powershell() -> Optional[str]:
    found = shutil.which("powershell") or shutil.which("powershell.exe")
    if found:
        return found
    candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return str(candidate) if candidate.is_file() else None


def parse_picker_output(raw: bytes) -> Optional[str]:
    """Interpreta a saída do PowerShell (UTF-8, possivelmente com BOM). Vazio = cancelado."""
    text = raw.decode("utf-8-sig", errors="replace").strip().strip("\r\n").strip()
    return text or None


def pick_file(
    title: str = "Selecione o vídeo ou áudio para transcrever",
    initial_dir: Optional[str | Path] = None,
    *,
    timeout: float = 900.0,
) -> Optional[str]:
    """Abre o explorador de arquivos e retorna o caminho escolhido (``None`` se cancelado)."""
    if not IS_WINDOWS:
        raise FilePickerError("O seletor de arquivos nativo só está disponível no Windows. Digite o caminho do arquivo.")
    exe = _find_powershell()
    if not exe:
        raise FilePickerError(technical="powershell.exe não encontrado")

    start_dir = str(initial_dir) if initial_dir and Path(initial_dir).is_dir() else None
    script = build_powershell_script(title, start_dir)
    args = [exe, "-NoProfile", "-NonInteractive", "-STA", "-ExecutionPolicy", "Bypass",
            "-EncodedCommand", encode_command(script)]
    try:
        result = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise FilePickerError(
            "O seletor de arquivos ficou aberto por muito tempo e foi fechado. Tente novamente.",
            technical=repr(exc),
        ) from exc
    except OSError as exc:
        raise FilePickerError(technical=repr(exc)) from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-1500:]
        logger.warning("Falha no seletor de arquivos (código %s): %s", result.returncode, stderr)
        raise FilePickerError(technical=stderr)
    return parse_picker_output(result.stdout)
