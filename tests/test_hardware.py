"""Testes de src/hardware.py: identificação de GPU/CUDA e fallback para CPU."""

from __future__ import annotations

import sys
import types

import pytest

import src.hardware as hw
from src.utils import GPUError

GB = 1024 ** 3


def make_hw(*, cuda=False, free_mb=8000, ram_gb=16.0, cores=8, cpu_types=("float32", "int8"),
            cuda_types=("float16", "int8_float16", "int8", "float32"), vendor="NVIDIA") -> hw.HardwareInfo:
    gpus = []
    if vendor == "NVIDIA" and cuda:
        gpus = [hw.GPUDevice(0, "NVIDIA RTX Teste", "NVIDIA", memory_total_mb=8192, memory_free_mb=free_mb,
                             driver_version="555.00", cuda_capable=True)]
    elif vendor == "AMD":
        gpus = [hw.GPUDevice(0, "AMD Radeon RX 580", "AMD", memory_total_mb=4096, cuda_capable=False)]
    status = hw.CudaStatus(
        ctranslate2_installed=True, cuda_device_count=1 if cuda else 0,
        compute_types=list(cuda_types) if cuda else [], cublas_ok=cuda, cudnn_ok=cuda, usable=cuda,
        reason="ok" if cuda else "nenhuma GPU NVIDIA com driver CUDA foi encontrada.",
    )
    return hw.HardwareInfo(
        os_name="Windows 10", python_version="3.11.9", cpu_name="CPU de teste",
        cpu_physical_cores=cores, cpu_logical_cores=cores * 2,
        ram_total_bytes=int(ram_gb * GB), ram_available_bytes=int(ram_gb * GB),
        gpus=gpus, cuda=status, cpu_compute_types=list(cpu_types),
    )


# ---------------------------------------------------------------------------
# 11. Escolha do dispositivo e fallback para CPU
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_cpu_requested_uses_int8(self):
        d = hw.resolve_device("cpu", "small", make_hw(cuda=True))
        assert (d.device, d.compute_type, d.fallback_used) == ("cpu", "int8", False)

    def test_cpu_without_int8_support_uses_float32(self):
        d = hw.resolve_device("cpu", "small", make_hw(cpu_types=("float32",)))
        assert d.compute_type == "float32"

    def test_auto_without_cuda_is_cpu_and_not_a_fallback(self):
        d = hw.resolve_device("auto", "small", make_hw(cuda=False))
        assert d.device == "cpu" and not d.fallback_used

    def test_auto_with_cuda_uses_gpu_float16(self):
        d = hw.resolve_device("auto", "small", make_hw(cuda=True, free_mb=7000))
        assert (d.device, d.compute_type) == ("cuda", "float16")
        assert d.device_index == 0 and not d.fallback_used

    def test_cuda_requested_but_unavailable_falls_back_to_cpu_with_warning(self):
        d = hw.resolve_device("cuda", "small", make_hw(cuda=False))
        assert d.device == "cpu" and d.fallback_used
        assert any("GPU" in w for w in d.warnings)

    def test_cuda_requested_but_unavailable_raises_when_fallback_forbidden(self):
        with pytest.raises(GPUError) as exc:
            hw.resolve_device("cuda", "small", make_hw(cuda=False), allow_cpu_fallback=False)
        assert "CPU" in exc.value.user_message  # a interface oferece a CPU

    def test_amd_gpu_is_never_used(self):
        d = hw.resolve_device("auto", "small", make_hw(vendor="AMD"))
        assert d.device == "cpu"

    def test_gpu_with_little_memory_reduces_precision(self):
        # small: float16 pede 1.2+0.7 GB; com 1.7 GB livres cai para int8_float16 (0.8+0.7)
        d = hw.resolve_device("auto", "small", make_hw(cuda=True, free_mb=1700))
        assert (d.device, d.compute_type) == ("cuda", "int8_float16")
        assert any("limitada" in w for w in d.warnings)

    def test_gpu_without_enough_memory_falls_back_to_cpu(self):
        d = hw.resolve_device("auto", "large-v3", make_hw(cuda=True, free_mb=700))
        assert d.device == "cpu" and d.fallback_used

    def test_gpu_without_enough_memory_is_an_error_if_explicit_and_no_fallback(self):
        with pytest.raises(GPUError):
            hw.resolve_device("cuda", "large-v3", make_hw(cuda=True, free_mb=700), allow_cpu_fallback=False)

    def test_low_ram_warning(self):
        d = hw.resolve_device("cpu", "large-v3", make_hw(ram_gb=2.0))
        assert any("Pouca RAM" in w for w in d.warnings)
        assert any("lento" in w for w in d.warnings)

    def test_unknown_device(self):
        with pytest.raises(ValueError):
            hw.resolve_device("tpu", "small", make_hw())

    @pytest.mark.parametrize("cores,threads", [(2, 2), (4, 3), (8, 7), (32, 16)])
    def test_cpu_threads_leave_one_core_free(self, cores, threads):
        assert hw.resolve_device("cpu", "base", make_hw(cores=cores)).cpu_threads == threads

    def test_cpu_fallback_decision(self):
        d = hw.cpu_fallback_decision("small", "CUDA out of memory")
        assert d.device == "cpu" and d.fallback_used
        assert any("GPU" in w for w in d.warnings)

    def test_decision_roundtrip(self):
        d = hw.resolve_device("auto", "small", make_hw(cuda=True))
        assert hw.DeviceDecision.from_dict(d.to_dict()) == d
        assert "GPU" in d.label


class TestHelpers:
    @pytest.mark.parametrize("exc,expected", [
        (RuntimeError("CUDA out of memory"), True),
        (RuntimeError("Library cublas64_12.dll is not found"), True),
        (RuntimeError("cuDNN error"), True),
        (ValueError("valor inválido"), False),
        (OSError("arquivo não encontrado"), False),
    ])
    def test_is_cuda_runtime_error(self, exc, expected):
        assert hw.is_cuda_runtime_error(exc) is expected

    def test_recommend_model(self):
        assert hw.recommend_model(make_hw(cuda=True, free_mb=10000)) == "large-v3"
        assert hw.recommend_model(make_hw(cuda=True, free_mb=4000)) == "medium"
        assert hw.recommend_model(make_hw(cuda=True, free_mb=1500)) == "small"
        assert hw.recommend_model(make_hw(ram_gb=1.0)) == "base"
        assert hw.recommend_model(make_hw(ram_gb=16.0)) == "small"

    def test_hardware_info_roundtrip_and_summary(self):
        info = make_hw(cuda=True)
        again = hw.HardwareInfo.from_dict(info.to_dict())
        assert again == info
        text = "\n".join(info.summary_lines())
        assert "CPU de teste" in text and "RTX" in text and "CUDA" in text

    def test_summary_for_amd_explains_why_no_cuda(self):
        text = "\n".join(make_hw(vendor="AMD").summary_lines())
        assert "Radeon" in text and "indisponível" in text


# ---------------------------------------------------------------------------
# 10. Identificação de GPU/CUDA (CTranslate2 simulado)
# ---------------------------------------------------------------------------


def fake_ctranslate2(count: int, compute=("float16", "int8")):
    mod = types.ModuleType("ctranslate2")
    mod.__version__ = "4.5.0"
    mod.get_cuda_device_count = lambda: count
    mod.get_supported_compute_types = lambda dev: list(compute) if dev == "cuda" else ["float32", "int8"]
    return mod


class TestCheckCudaSupport:
    def test_ctranslate2_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ctranslate2", None)
        st = hw.check_cuda_support()
        assert not st.usable and not st.ctranslate2_installed and "CTranslate2" in st.reason

    def test_no_cuda_device(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2(0))
        st = hw.check_cuda_support()
        assert not st.usable and st.cuda_device_count == 0 and "nenhuma GPU" in st.reason

    def test_device_present_but_cublas_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2(1))
        monkeypatch.setattr(hw, "_can_load_libs", lambda names: False)
        st = hw.check_cuda_support()
        assert not st.usable and "cuBLAS" in st.reason

    def test_cublas_ok_cudnn_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2(1))
        monkeypatch.setattr(hw, "_can_load_libs", lambda names: any("cublas" in n.lower() for n in names))
        st = hw.check_cuda_support()
        assert not st.usable and st.cublas_ok and not st.cudnn_ok and "cuDNN" in st.reason

    def test_fully_usable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2(1))
        monkeypatch.setattr(hw, "_can_load_libs", lambda names: True)
        st = hw.check_cuda_support()
        assert st.usable and "float16" in st.compute_types and st.ctranslate2_version == "4.5.0"


class TestRealDetection:
    def test_detect_hardware_returns_sane_values(self):
        info = hw.detect_hardware()
        assert info.cpu_logical_cores >= 1 and info.cpu_physical_cores >= 1
        assert info.ram_total_bytes > 0 and 0 < info.ram_available_bytes <= info.ram_total_bytes
        assert info.cpu_name
        assert info.summary_lines()

    def test_decision_always_possible_on_this_machine(self):
        d = hw.resolve_device("auto", "base", hw.detect_hardware())
        assert d.device in ("cpu", "cuda") and d.compute_type

    def test_isolated_probe_matches_json_contract(self):
        info = hw.probe_hardware_subprocess(timeout=120)
        assert isinstance(info, hw.HardwareInfo) and info.cpu_logical_cores >= 1

    def test_nvidia_smi_absent_returns_empty(self, monkeypatch):
        monkeypatch.setattr(hw.shutil, "which", lambda name: None)
        monkeypatch.setattr(hw, "IS_WINDOWS", False)
        assert hw.query_nvidia_gpus() == []
