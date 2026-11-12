"""Testes de src/transcription.py com um faster-whisper simulado (não baixa modelos nem usa GPU)."""

from __future__ import annotations

import errno
from types import SimpleNamespace

import pytest

import src.transcription as tr
import src.utils as u
from src.hardware import DeviceDecision
from src.subtitles import Segment
from src.transcription import TranscriptionEngine, TranscriptionSettings
from src.utils import (
    DiskSpaceError,
    GPUError,
    InsufficientMemoryError,
    ModelLoadError,
    TaskCancelledError,
    TranscriberError,
)

CPU = DeviceDecision(device="cpu", compute_type="int8", cpu_threads=4)
GPU = DeviceDecision(device="cuda", compute_type="float16", cpu_threads=2, device_index=0)


def raw(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


class FakeModel:
    def __init__(self, segments=(), language="pt", probability=0.98, error=None):
        self.segments = list(segments)
        self.language, self.probability, self.error = language, probability, error
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if self.error is not None:
            raise self.error
        return iter(self.segments), SimpleNamespace(language=self.language, language_probability=self.probability)


class Factory:
    """Entrega, a cada chamada, o próximo item da lista (exceção → é lançada)."""

    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def __call__(self, *args):
        self.calls.append(args)
        item = self.items[min(len(self.calls) - 1, len(self.items) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture(autouse=True)
def plenty_of_ram(monkeypatch):
    monkeypatch.setattr(tr, "get_available_ram", lambda: 64 * 1024 ** 3)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(u, "interruptible_sleep", lambda *a, **k: None)


def engine(factory, decision=CPU, **kw):
    return TranscriptionEngine(TranscriptionSettings(model="small"), decision, model_factory=factory, **kw)


# ---------------------------------------------------------------------------
# Parâmetros do faster-whisper
# ---------------------------------------------------------------------------


class TestSettingsAndKwargs:
    def test_defaults_follow_the_spec(self):
        s = TranscriptionSettings()
        assert s.model == "small" and s.language == "pt" and s.vad_filter and s.beam_size == 5
        assert s.condition_on_previous_text is False  # reduz repetições/alucinações

    def test_kwargs_with_vad(self):
        kw = tr.build_transcribe_kwargs(TranscriptionSettings(), "pt")
        assert kw["vad_filter"] is True and kw["language"] == "pt" and kw["task"] == "transcribe"
        assert kw["vad_parameters"]["min_silence_duration_ms"] == 500
        assert kw["vad_parameters"]["speech_pad_ms"] >= 300  # preserva pausas/bordas de palavras
        assert kw["condition_on_previous_text"] is False
        assert kw["word_timestamps"] is False
        assert kw["compression_ratio_threshold"] == 2.4 and kw["log_prob_threshold"] == -1.0
        assert isinstance(kw["temperature"], list) and kw["temperature"][0] == 0.0

    def test_kwargs_without_vad(self):
        kw = tr.build_transcribe_kwargs(TranscriptionSettings(vad_filter=False), "pt")
        assert kw["vad_filter"] is False and "vad_parameters" not in kw

    def test_auto_language_and_beam_floor(self):
        kw = tr.build_transcribe_kwargs(TranscriptionSettings(beam_size=0), "auto")
        assert kw["language"] is None and kw["beam_size"] == 1

    @pytest.mark.parametrize("value,expected", [("auto", None), ("", None), (None, None), (" PT ", "pt"), ("en", "en")])
    def test_normalize_language(self, value, expected):
        assert tr.normalize_language(value) == expected

    def test_serialization(self):
        s = TranscriptionSettings(model="medium", beam_size=3, temperature=(0.0, 0.5))
        again = TranscriptionSettings.from_dict(s.to_dict())
        assert again == s and isinstance(again.temperature, tuple)
        assert TranscriptionSettings.from_dict({"model": "base", "campo_desconhecido": 1}).model == "base"
        assert TranscriptionSettings.from_dict(None) == TranscriptionSettings()

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            TranscriptionEngine(TranscriptionSettings(model="gigante"), CPU)


# ---------------------------------------------------------------------------
# 7. Conversão de timestamps
# ---------------------------------------------------------------------------


class TestTimestampConversion:
    def test_offset_is_added(self):
        s = tr.make_segment(1.5, 4.0, " texto ", offset=600, chunk_duration=600)
        assert (s.start, s.end, s.text) == (601.5, 604.0, "texto")

    def test_clamped_to_chunk_bounds(self):
        s = tr.make_segment(-1, 700, "x", offset=1200, chunk_duration=600)
        assert (s.start, s.end) == (1200, 1800)

    def test_end_never_before_start(self):
        s = tr.make_segment(10, 5, "x", offset=0, chunk_duration=600)
        assert s.end == s.start == 10

    def test_empty_text(self):
        assert tr.make_segment(0, 1, "   ", 0, 600) is None
        assert tr.make_segment(0, 1, None, 0, 600) is None

    def test_convert_raw_segments(self):
        out = tr.convert_raw_segments([raw(0, 1, "a"), raw(1, 2, " "), raw(2, 3, "b")], offset=100, chunk_duration=600)
        assert [(s.start, s.text) for s in out] == [(100, "a"), (102, "b")]

    def test_two_consecutive_chunks_form_a_continuous_timeline(self):
        c1 = tr.convert_raw_segments([raw(0, 3, "a"), raw(590, 600, "b")], offset=0, chunk_duration=600)
        c2 = tr.convert_raw_segments([raw(0, 4, "c")], offset=600, chunk_duration=583.5)
        allsegs = c1 + c2
        assert all(a.start <= b.start for a, b in zip(allsegs, allsegs[1:]))
        assert c1[-1].end <= c2[0].start


# ---------------------------------------------------------------------------
# Motor: transcrição de um bloco
# ---------------------------------------------------------------------------


class TestTranscribeFile:
    def test_segments_get_original_video_time(self):
        model = FakeModel([raw(0.0, 5.0, "Olá"), raw(5.0, 9.5, " mundo "), raw(9.5, 9.7, "   ")])
        eng = engine(Factory(model))
        progress = []
        res = eng.transcribe_file("bloco.wav", offset=600.0, chunk_duration=600.0, language="pt",
                                  progress_cb=lambda s, d: progress.append((s, d)))
        assert res.segments == [Segment(600.0, 605.0, "Olá"), Segment(605.0, 609.5, "mundo")]
        assert res.language == "pt" and res.language_probability == pytest.approx(0.98)
        assert res.speech_detected and res.realtime_factor is not None
        assert progress[-1] == (600.0, 600.0)
        assert [p[0] for p in progress] == sorted(p[0] for p in progress)
        path, kwargs = model.calls[0]
        assert path == "bloco.wav" and kwargs["language"] == "pt" and kwargs["vad_filter"] is True

    def test_end_beyond_chunk_is_clamped(self):
        eng = engine(Factory(FakeModel([raw(590, 650, "fim")])))
        res = eng.transcribe_file("b.wav", offset=600, chunk_duration=600, language=None)
        assert (res.segments[0].start, res.segments[0].end) == (1190, 1200)

    def test_silence_returns_no_segments_and_no_invented_text(self):
        res = engine(Factory(FakeModel([]))).transcribe_file("b.wav", offset=0, chunk_duration=600, language="pt")
        assert res.segments == [] and not res.speech_detected

    def test_repeated_phrases_are_kept(self):
        segs = [raw(0, 1, "sim"), raw(1, 2, "sim"), raw(2, 3, "sim")]
        res = engine(Factory(FakeModel(segs))).transcribe_file("b.wav", offset=0, chunk_duration=60, language="pt")
        assert [s.text for s in res.segments] == ["sim", "sim", "sim"]

    def test_auto_language_passes_none_and_reports_detection(self):
        model = FakeModel([raw(0, 1, "hello")], language="en", probability=0.91)
        res = engine(Factory(model)).transcribe_file("b.wav", offset=0, chunk_duration=60, language=None)
        assert model.calls[0][1]["language"] is None and res.language == "en"

    def test_cancel_between_segments(self):
        model = FakeModel([raw(0, 1, "a"), raw(1, 2, "b"), raw(2, 3, "c")])
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] >= 2

        with pytest.raises(TaskCancelledError):
            engine(Factory(model)).transcribe_file("b.wav", offset=0, chunk_duration=60, language="pt", cancel_check=cancel)

    def test_generic_failure_becomes_friendly_error_with_technical_detail(self):
        eng = engine(Factory(FakeModel(error=ValueError("boom"))))
        with pytest.raises(TranscriberError) as exc:
            eng.transcribe_file("b.wav", offset=0, chunk_duration=60, language="pt")
        assert "boom" in (exc.value.technical or "") and exc.value.user_message

    def test_memory_error_during_transcription(self):
        eng = engine(Factory(FakeModel(error=MemoryError())))
        with pytest.raises(InsufficientMemoryError):
            eng.transcribe_file("b.wav", offset=0, chunk_duration=60, language="pt")

    def test_close_releases_the_model(self):
        eng = engine(Factory(FakeModel()))
        eng.load()
        assert eng.is_loaded
        eng.close()
        assert not eng.is_loaded


# ---------------------------------------------------------------------------
# Carregamento do modelo
# ---------------------------------------------------------------------------


class TestLoad:
    def test_factory_receives_device_settings(self, tmp_path):
        f = Factory(FakeModel())
        TranscriptionEngine(TranscriptionSettings(model="small"), CPU, model_factory=f, download_root=tmp_path).load()
        assert f.calls == [("small", "cpu", 0, "int8", 4, str(tmp_path))]

    def test_load_is_idempotent(self):
        f = Factory(FakeModel())
        eng = engine(f)
        eng.load()
        eng.load()
        assert len(f.calls) == 1

    def test_network_failures_are_retried(self, no_sleep):
        f = Factory(ConnectionError("Connection timed out"), ConnectionError("Connection reset by peer"), FakeModel())
        messages = []
        eng = engine(f, on_message=messages.append)
        eng.load()
        assert len(f.calls) == 3 and eng.is_loaded
        assert any("Falha de conexão" in m for m in messages)

    def test_network_failure_after_all_attempts(self, no_sleep):
        f = Factory(ConnectionError("Connection timed out"))
        with pytest.raises(ModelLoadError):
            engine(f).load()
        assert len(f.calls) == 3

    def test_other_errors_are_not_retried(self, no_sleep):
        f = Factory(ValueError("modelo corrompido"))
        with pytest.raises(ModelLoadError) as exc:
            engine(f).load()
        assert len(f.calls) == 1 and "modelo corrompido" in (exc.value.technical or "")

    def test_memory_error(self):
        with pytest.raises(InsufficientMemoryError):
            engine(Factory(MemoryError())).load()

    def test_disk_full_while_downloading_model(self):
        with pytest.raises(DiskSpaceError):
            engine(Factory(OSError(errno.ENOSPC, "No space left on device"))).load()

    def test_not_enough_ram_fails_before_loading(self, monkeypatch):
        monkeypatch.setattr(tr, "get_available_ram", lambda: 100 * 1024 ** 2)
        f = Factory(FakeModel())
        with pytest.raises(InsufficientMemoryError) as exc:
            engine(f).load()
        assert f.calls == [] and "RAM" in exc.value.user_message

    def test_looks_like_network_error(self):
        assert tr.looks_like_network_error(ConnectionError("x"))
        assert tr.looks_like_network_error(OSError("Name resolution failed"))
        assert not tr.looks_like_network_error(ValueError("arquivo inválido"))


# ---------------------------------------------------------------------------
# 11. Fallback GPU → CPU
# ---------------------------------------------------------------------------


class TestGpuFallback:
    def test_load_failure_on_gpu_falls_back_to_cpu(self):
        f = Factory(RuntimeError("CUDA failed to initialize cublas"), FakeModel([raw(0, 1, "ok")]))
        changes = []
        eng = engine(f, GPU, allow_cpu_fallback=True, on_device_change=lambda d, r: changes.append((d, r)))
        eng.load()
        assert eng.decision.device == "cpu" and eng.decision.fallback_used
        assert f.calls[0][1] == "cuda" and f.calls[1][1] == "cpu"
        assert len(changes) == 1 and changes[0][0].device == "cpu"

    def test_load_failure_on_gpu_raises_when_fallback_not_allowed(self):
        eng = engine(Factory(RuntimeError("CUDA failed")), GPU, allow_cpu_fallback=False)
        with pytest.raises(GPUError) as exc:
            eng.load()
        assert "CPU" in exc.value.user_message

    def test_runtime_gpu_failure_continues_on_cpu(self):
        gpu_model = FakeModel(error=RuntimeError("CUDA out of memory"))
        cpu_model = FakeModel([raw(0, 2, "continuou na CPU")])
        eng = engine(Factory(gpu_model, cpu_model), GPU, allow_cpu_fallback=True)
        res = eng.transcribe_file("b.wav", offset=60, chunk_duration=60, language="pt")
        assert [s.text for s in res.segments] == ["continuou na CPU"] and res.segments[0].start == 60
        assert eng.decision.device == "cpu"

    def test_runtime_gpu_failure_without_fallback(self):
        eng = engine(Factory(FakeModel(error=RuntimeError("CUDA out of memory"))), GPU, allow_cpu_fallback=False)
        with pytest.raises(GPUError):
            eng.transcribe_file("b.wav", offset=0, chunk_duration=60, language="pt")

    def test_cpu_engine_never_switches_device(self):
        eng = engine(Factory(FakeModel(error=RuntimeError("CUDA out of memory"))), CPU)
        with pytest.raises(TranscriberError):
            eng.transcribe_file("b.wav", offset=0, chunk_duration=60, language="pt")
        assert eng.decision.device == "cpu"
