"""Detecção de hardware e escolha do dispositivo de transcrição.

Responsabilidades:

* Informar processador, RAM, GPU e memória da GPU (sem dependências extras:
  usa ``psutil``, ``winreg``, ``nvidia-smi`` e consultas nativas do Windows).
* Verificar se o **CTranslate2** consegue de fato usar CUDA — não basta haver
  uma placa de vídeo: é preciso GPU NVIDIA, driver, CUDA 12 (cuBLAS) e cuDNN 9.
* Registrar as pastas das DLLs CUDA instaladas via pip
  (``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12``) para o Windows encontrá-las.
* Decidir dispositivo (``cpu``/``cuda``), tipo de computação (``float16``,
  ``int8_float16``, ``int8``...) e número de threads, com fallback para CPU.

Pode ser executado como processo isolado — ``python -m src.hardware --json`` —
para que a interface não precise carregar DLLs CUDA no próprio processo.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import psutil

from .utils import CREATE_NO_WINDOW, IS_WINDOWS, MODELS, GPUError, format_bytes

logger = logging.getLogger(__name__)

GB = 1024 ** 3

#: DLLs necessárias ao CTranslate2 4.x com CUDA 12 / cuDNN 9.
_CUBLAS_LIBS_WIN = ("cublas64_12.dll", "cublasLt64_12.dll")
_CUDNN_LIBS_WIN = ("cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll")
_CUBLAS_LIBS_LINUX = ("libcublas.so.12", "libcublasLt.so.12")
_CUDNN_LIBS_LINUX = ("libcudnn.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9")

_dll_dir_handles: list[Any] = []  # mantém as referências vivas (senão o diretório é removido)
_dll_dirs_registered = False


# =============================================================================
# Estruturas de dados
# =============================================================================


@dataclass
class GPUDevice:
    index: int
    name: str
    vendor: str = "desconhecido"
    memory_total_mb: Optional[int] = None
    memory_free_mb: Optional[int] = None
    driver_version: Optional[str] = None
    cuda_capable: bool = False


@dataclass
class CudaStatus:
    ctranslate2_installed: bool = False
    ctranslate2_version: Optional[str] = None
    cuda_device_count: int = 0
    compute_types: list[str] = field(default_factory=list)
    cublas_ok: bool = False
    cudnn_ok: bool = False
    usable: bool = False
    reason: str = ""


@dataclass
class HardwareInfo:
    os_name: str
    python_version: str
    cpu_name: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    ram_total_bytes: int
    ram_available_bytes: int
    gpus: list[GPUDevice]
    cuda: CudaStatus
    cpu_compute_types: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardwareInfo":
        return cls(
            os_name=data["os_name"],
            python_version=data["python_version"],
            cpu_name=data["cpu_name"],
            cpu_physical_cores=int(data["cpu_physical_cores"]),
            cpu_logical_cores=int(data["cpu_logical_cores"]),
            ram_total_bytes=int(data["ram_total_bytes"]),
            ram_available_bytes=int(data["ram_available_bytes"]),
            gpus=[GPUDevice(**g) for g in data.get("gpus", [])],
            cuda=CudaStatus(**data.get("cuda", {})),
            cpu_compute_types=list(data.get("cpu_compute_types", [])),
        )

    @property
    def primary_cuda_gpu(self) -> Optional[GPUDevice]:
        return next((g for g in self.gpus if g.cuda_capable), None)

    def summary_lines(self) -> list[str]:
        """Linhas prontas para exibição na interface."""
        lines = [
            f"Processador: {self.cpu_name} ({self.cpu_physical_cores} núcleos / {self.cpu_logical_cores} threads)",
            f"Memória RAM: {format_bytes(self.ram_available_bytes)} livres de {format_bytes(self.ram_total_bytes)}",
        ]
        if self.gpus:
            for gpu in self.gpus:
                mem = ""
                if gpu.memory_total_mb:
                    mem = f" — {gpu.memory_total_mb / 1024:.1f} GB"
                    if gpu.memory_free_mb is not None:
                        mem += f" ({gpu.memory_free_mb / 1024:.1f} GB livres)"
                compat = "compatível com CUDA" if gpu.cuda_capable else "sem suporte CUDA"
                lines.append(f"GPU: {gpu.name}{mem} [{compat}]")
        else:
            lines.append("GPU: nenhuma identificada")
        lines.append(
            "Aceleração CUDA: disponível" if self.cuda.usable else f"Aceleração CUDA: indisponível — {self.cuda.reason}"
        )
        return lines


@dataclass
class DeviceDecision:
    """Resultado da escolha de dispositivo para o faster-whisper."""

    device: str  # "cpu" ou "cuda"
    compute_type: str
    cpu_threads: int
    device_index: int = 0
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        name = "GPU (CUDA)" if self.device == "cuda" else "CPU"
        return f"{name} · {self.compute_type}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceDecision":
        return cls(**data)


# =============================================================================
# DLLs CUDA (Windows)
# =============================================================================


def _candidate_cuda_dll_dirs() -> list[Path]:
    """Pastas ``site-packages/nvidia/*/bin`` criadas pelos pacotes pip da NVIDIA."""
    dirs: list[Path] = []
    roots: list[Path] = []
    for entry in sys.path:
        try:
            p = Path(entry)
        except (TypeError, ValueError):
            continue
        if p.name in ("site-packages", "dist-packages") and p.is_dir():
            roots.append(p)
    for root in roots:
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            for sub in nvidia.iterdir():
                for leaf in ("bin", "lib"):
                    candidate = sub / leaf
                    if candidate.is_dir():
                        dirs.append(candidate)
    # Instalação do CUDA Toolkit oficial (opcional).
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and Path(cuda_path, "bin").is_dir():
        dirs.append(Path(cuda_path, "bin"))
    return dirs


def register_cuda_dll_dirs() -> list[str]:
    """Torna as DLLs CUDA/cuDNN visíveis para o processo atual (idempotente).

    Deve ser chamada ANTES de importar/carregar o modelo do faster-whisper.
    """
    global _dll_dirs_registered
    if _dll_dirs_registered or not IS_WINDOWS:
        return []
    added: list[str] = []
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for d in _candidate_cuda_dll_dirs():
        ds = str(d)
        try:
            _dll_dir_handles.append(os.add_dll_directory(ds))  # type: ignore[attr-defined]
        except (OSError, AttributeError):
            continue
        if ds not in path_parts:
            path_parts.insert(0, ds)
        added.append(ds)
    if added:
        # O CTranslate2 carrega o cuDNN sob demanda via LoadLibrary, que também consulta o PATH.
        os.environ["PATH"] = os.pathsep.join(path_parts)
        logger.info("Pastas de DLL CUDA registradas: %s", added)
    _dll_dirs_registered = True
    return added


def _can_load_libs(names: tuple[str, ...]) -> bool:
    for name in names:
        try:
            if IS_WINDOWS:
                ctypes.WinDLL(name, winmode=0)  # winmode=0 → busca padrão (inclui PATH)
            else:
                ctypes.CDLL(name)
        except OSError:
            return False
    return True


# =============================================================================
# CPU / RAM
# =============================================================================


def get_cpu_name() -> str:
    if IS_WINDOWS:
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return " ".join(str(value).split())
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "Processador desconhecido"


def get_cpu_cores() -> tuple[int, int]:
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    return int(physical), int(logical)


def recommended_cpu_threads() -> int:
    """Threads para o CTranslate2: núcleos físicos, deixando 1 livre para a interface."""
    physical, _ = get_cpu_cores()
    threads = physical - 1 if physical >= 4 else physical
    return max(1, min(threads, 16))


# =============================================================================
# GPU
# =============================================================================


def _run_quiet(args: list[str], timeout: float = 10.0) -> Optional[str]:
    try:
        out = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def query_nvidia_gpus() -> list[GPUDevice]:
    """GPUs NVIDIA via ``nvidia-smi`` (instalado junto com o driver)."""
    exe = shutil.which("nvidia-smi")
    if not exe and IS_WINDOWS:
        fallback = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(fallback) if fallback.exists() else None
    if not exe:
        return []
    text = _run_quiet(
        [exe, "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"]
    )
    gpus: list[GPUDevice] = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                GPUDevice(
                    index=int(parts[0]),
                    name=parts[1],
                    vendor="NVIDIA",
                    memory_total_mb=int(float(parts[2])),
                    memory_free_mb=int(float(parts[3])),
                    driver_version=parts[4],
                    cuda_capable=True,
                )
            )
        except ValueError:
            continue
    return gpus


def query_other_gpus() -> list[GPUDevice]:
    """Placas de vídeo não-NVIDIA (apenas informativo) via consulta nativa do Windows."""
    if not IS_WINDOWS:
        return []
    ps = shutil.which("powershell") or shutil.which("powershell.exe")
    if not ps:
        return []
    text = _run_quiet(
        [
            ps, "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)\" }",
        ],
        timeout=15,
    )
    gpus: list[GPUDevice] = []
    for i, line in enumerate((text or "").splitlines()):
        if "|" not in line:
            continue
        name, ram = line.rsplit("|", 1)
        name = name.strip()
        if not name or "nvidia" in name.lower():
            continue
        lower = name.lower()
        vendor = "AMD" if ("amd" in lower or "radeon" in lower) else "Intel" if "intel" in lower else "outro"
        mem_mb: Optional[int] = None
        try:
            # AdapterRAM é limitado a 4 GB (uint32) pelo Windows; valor apenas indicativo.
            mem_mb = int(int(ram.strip()) / (1024 * 1024)) or None
        except ValueError:
            pass
        gpus.append(GPUDevice(index=i, name=name, vendor=vendor, memory_total_mb=mem_mb, cuda_capable=False))
    return gpus


def check_cuda_support() -> CudaStatus:
    """Verifica se o CTranslate2 instalado consegue usar CUDA neste computador."""
    status = CudaStatus()
    register_cuda_dll_dirs()
    try:
        import ctranslate2  # noqa: WPS433 (import tardio proposital)
    except Exception as exc:  # noqa: BLE001 - ImportError ou falha de DLL
        status.reason = f"CTranslate2 não pôde ser carregado ({type(exc).__name__})."
        return status
    status.ctranslate2_installed = True
    status.ctranslate2_version = getattr(ctranslate2, "__version__", None)

    try:
        status.cuda_device_count = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:  # noqa: BLE001
        status.reason = f"Falha ao consultar CUDA: {exc}"
        return status
    if status.cuda_device_count <= 0:
        status.reason = "nenhuma GPU NVIDIA com driver CUDA foi encontrada."
        return status

    try:
        status.compute_types = sorted(ctranslate2.get_supported_compute_types("cuda"))
    except Exception as exc:  # noqa: BLE001
        status.reason = f"GPU encontrada, mas sem tipos de computação suportados ({exc})."
        return status

    cublas = _CUBLAS_LIBS_WIN if IS_WINDOWS else _CUBLAS_LIBS_LINUX
    cudnn = _CUDNN_LIBS_WIN if IS_WINDOWS else _CUDNN_LIBS_LINUX
    status.cublas_ok = _can_load_libs(cublas)
    status.cudnn_ok = _can_load_libs(cudnn)
    if not status.cublas_ok:
        status.reason = "biblioteca cuBLAS (CUDA 12) não encontrada. Instale requirements-gpu.txt."
        return status
    if not status.cudnn_ok:
        status.reason = "biblioteca cuDNN 9 não encontrada. Instale requirements-gpu.txt."
        return status

    status.usable = True
    status.reason = "ok"
    return status


def get_cpu_compute_types() -> list[str]:
    try:
        import ctranslate2

        return sorted(ctranslate2.get_supported_compute_types("cpu"))
    except Exception:  # noqa: BLE001
        return ["float32", "int8"]


# =============================================================================
# Detecção completa
# =============================================================================


def detect_hardware() -> HardwareInfo:
    """Coleta todas as informações de hardware relevantes."""
    physical, logical = get_cpu_cores()
    vm = psutil.virtual_memory()
    cuda = check_cuda_support()
    gpus = query_nvidia_gpus()
    if cuda.usable and not gpus:
        # Driver sem nvidia-smi no PATH: registra GPU genérica.
        gpus = [GPUDevice(index=i, name=f"GPU NVIDIA #{i}", vendor="NVIDIA", cuda_capable=True)
                for i in range(cuda.cuda_device_count)]
    gpus += query_other_gpus()
    if not cuda.usable:
        for gpu in gpus:
            if gpu.vendor == "NVIDIA" and cuda.cuda_device_count == 0:
                gpu.cuda_capable = False
        if any(g.vendor in ("AMD", "Intel") for g in gpus) and not any(g.vendor == "NVIDIA" for g in gpus):
            cuda.reason = "a GPU instalada não é NVIDIA; o CTranslate2 só acelera com CUDA (NVIDIA)."
    return HardwareInfo(
        os_name=f"{platform.system()} {platform.release()} ({platform.version()})",
        python_version=platform.python_version(),
        cpu_name=get_cpu_name(),
        cpu_physical_cores=physical,
        cpu_logical_cores=logical,
        ram_total_bytes=int(vm.total),
        ram_available_bytes=int(vm.available),
        gpus=gpus,
        cuda=cuda,
        cpu_compute_types=get_cpu_compute_types(),
    )


def probe_hardware_subprocess(timeout: float = 90.0) -> HardwareInfo:
    """Executa a detecção em um processo separado (``python -m src.hardware --json``).

    Protege a interface: se uma DLL CUDA travar ou derrubar o processo, apenas o
    processo de diagnóstico é afetado. Em caso de falha, usa uma detecção local
    mínima (sem CTranslate2).
    """
    from . import PROJECT_ROOT

    try:
        out = subprocess.run(
            [sys.executable, "-m", "src.hardware", "--json"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW,
        )
        if out.returncode == 0:
            # A última linha não vazia contém o JSON (logs podem vir antes).
            payload = [ln for ln in out.stdout.splitlines() if ln.strip()][-1]
            return HardwareInfo.from_dict(json.loads(payload))
        logger.warning("Diagnóstico de hardware retornou %s: %s", out.returncode, out.stderr[-2000:])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError, KeyError) as exc:
        logger.warning("Falha no diagnóstico de hardware isolado: %r", exc)

    physical, logical = get_cpu_cores()
    vm = psutil.virtual_memory()
    return HardwareInfo(
        os_name=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        cpu_name=get_cpu_name(),
        cpu_physical_cores=physical,
        cpu_logical_cores=logical,
        ram_total_bytes=int(vm.total),
        ram_available_bytes=int(vm.available),
        gpus=query_nvidia_gpus() + query_other_gpus(),
        cuda=CudaStatus(reason="não foi possível verificar o CUDA neste momento."),
        cpu_compute_types=["float32", "int8"],
    )


# =============================================================================
# Escolha do dispositivo
# =============================================================================


def _pick_cpu(model_name: str, hw: HardwareInfo, warnings: list[str], fallback: bool) -> DeviceDecision:
    compute = "int8" if "int8" in hw.cpu_compute_types else "float32"
    info = MODELS.get(model_name)
    if info:
        needed = (info.ram_gb_int8 if compute == "int8" else info.ram_gb_int8 * 2.2) + 1.0
        if hw.ram_available_bytes < needed * GB:
            warnings.append(
                f"Pouca RAM livre ({format_bytes(hw.ram_available_bytes)}) para o modelo '{model_name}' "
                f"(recomendado ≈ {needed:.1f} GB). Feche outros programas ou use um modelo menor."
            )
        if model_name in ("medium", "large-v3"):
            warnings.append(
                f"O modelo '{model_name}' na CPU é lento: um vídeo de 4 horas pode levar muitas horas."
            )
    physical, _ = hw.cpu_physical_cores, hw.cpu_logical_cores
    threads = max(1, min(physical - 1 if physical >= 4 else physical, 16))
    return DeviceDecision(device="cpu", compute_type=compute, cpu_threads=threads, fallback_used=fallback, warnings=warnings)


def resolve_device(
    requested: str,
    model_name: str,
    hw: Optional[HardwareInfo] = None,
    *,
    allow_cpu_fallback: bool = True,
) -> DeviceDecision:
    """Decide dispositivo e precisão.

    * ``requested="auto"``: GPU quando utilizável e com memória suficiente; senão CPU.
    * ``requested="cuda"``: GPU; se indisponível, CPU com aviso — ou ``GPUError``
      quando ``allow_cpu_fallback=False`` (a interface pode então oferecer a CPU).
    * ``requested="cpu"``: CPU (``int8`` quando suportado).
    """
    hw = hw or detect_hardware()
    requested = (requested or "auto").lower()
    warnings: list[str] = []

    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Dispositivo desconhecido: {requested}")

    if requested == "cpu":
        return _pick_cpu(model_name, hw, warnings, fallback=False)

    if not hw.cuda.usable:
        if requested == "cuda":
            msg = f"A GPU/CUDA não está disponível: {hw.cuda.reason}"
            if not allow_cpu_fallback:
                raise GPUError(msg + " Deseja processar pela CPU?")
            warnings.append(msg + " A transcrição será feita pela CPU.")
            return _pick_cpu(model_name, hw, warnings, fallback=True)
        return _pick_cpu(model_name, hw, warnings, fallback=False)

    info = MODELS.get(model_name)
    gpu = hw.primary_cuda_gpu
    free_gb = (gpu.memory_free_mb / 1024) if gpu and gpu.memory_free_mb is not None else None
    types = set(hw.cuda.compute_types)
    margin = 0.7  # GB reservados para o driver/ativação

    compute: Optional[str] = None
    if info is None or free_gb is None:
        compute = "float16" if "float16" in types else ("int8_float16" if "int8_float16" in types else None)
    else:
        if "float16" in types and free_gb >= info.vram_gb_float16 + margin:
            compute = "float16"
        elif "int8_float16" in types and free_gb >= info.vram_gb_int8_float16 + margin:
            compute = "int8_float16"
            warnings.append("Memória da GPU limitada: usando precisão int8_float16 para caber na placa.")
        elif "int8" in types and free_gb >= info.vram_gb_int8_float16 + margin:
            compute = "int8"
            warnings.append("Memória da GPU limitada: usando precisão int8.")

    if compute is None:
        msg = (
            f"A GPU não tem memória livre suficiente para o modelo '{model_name}'"
            + (f" ({free_gb:.1f} GB livres)." if free_gb is not None else ".")
        )
        if requested == "cuda" and not allow_cpu_fallback:
            raise GPUError(msg + " Escolha um modelo menor ou processe pela CPU.")
        warnings.append(msg + " A transcrição será feita pela CPU.")
        return _pick_cpu(model_name, hw, warnings, fallback=True)

    return DeviceDecision(
        device="cuda",
        compute_type=compute,
        cpu_threads=min(4, hw.cpu_physical_cores),  # threads auxiliares (pré-processamento)
        device_index=gpu.index if gpu else 0,
        fallback_used=False,
        warnings=warnings,
    )


def cpu_fallback_decision(model_name: str, reason: str, hw: Optional[HardwareInfo] = None) -> DeviceDecision:
    """Decisão de CPU usada pelo worker quando a GPU falha em tempo de execução."""
    hw = hw or HardwareInfo(
        os_name="", python_version="", cpu_name="",
        cpu_physical_cores=get_cpu_cores()[0], cpu_logical_cores=get_cpu_cores()[1],
        ram_total_bytes=int(psutil.virtual_memory().total),
        ram_available_bytes=int(psutil.virtual_memory().available),
        gpus=[], cuda=CudaStatus(), cpu_compute_types=get_cpu_compute_types(),
    )
    return _pick_cpu(model_name, hw, [f"Falha na GPU ({reason}). Continuando pela CPU."], fallback=True)


def recommend_model(hw: HardwareInfo) -> str:
    """Sugestão de modelo conforme os recursos (o padrão do app continua 'small')."""
    gpu = hw.primary_cuda_gpu
    if hw.cuda.usable and gpu and gpu.memory_free_mb:
        free_gb = gpu.memory_free_mb / 1024
        if free_gb >= MODELS["large-v3"].vram_gb_float16 + 0.7:
            return "large-v3"
        if free_gb >= MODELS["medium"].vram_gb_float16 + 0.7:
            return "medium"
        return "small"
    ram_gb = hw.ram_available_bytes / GB
    if ram_gb < MODELS["small"].ram_gb_int8 + 1.0:
        return "base"
    return "small"


def is_cuda_runtime_error(exc: BaseException) -> bool:
    """Heurística: a exceção indica falha de CUDA/cuDNN/memória de GPU?"""
    text = f"{type(exc).__name__} {exc}".lower()
    keys = ("cuda", "cudnn", "cublas", "out of memory", "gpu", "device-side", "cudaerror")
    return any(k in text for k in keys)


# =============================================================================
# CLI de diagnóstico
# =============================================================================


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico de hardware do Transcritor")
    parser.add_argument("--json", action="store_true", help="imprime o resultado em JSON (uma linha)")
    args = parser.parse_args(argv)
    hw = detect_hardware()
    if args.json:
        # ensure_ascii=True: o stdout do processo filho usa a página de código do Windows (cp1252),
        # então acentos fora do ASCII viravam "�" no processo pai. Escapes \uXXXX são lidos corretamente.
        sys.stdout.write(json.dumps(hw.to_dict(), ensure_ascii=True) + "\n")
    else:
        for line in hw.summary_lines():
            print(line)
        decision = resolve_device("auto", "small", hw)
        print(f"Dispositivo sugerido (modelo small): {decision.label}, threads={decision.cpu_threads}")
        for w in decision.warnings:
            print(f"Aviso: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
