"""Testes de src/audio_processor.py: FFmpeg, extração, detecção de pausas e divisão em blocos."""

from __future__ import annotations

import array
import random
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path

import psutil
import pytest
from conftest import env_flag, sha256

import src.audio_processor as ap
from src.audio_processor import ChunkSpec, SilenceInterval, SilenceSettings
from src.utils import (
    AudioProcessingError,
    CorruptedMediaError,
    DiskSpaceError,
    FFmpegNotFoundError,
    TaskCancelledError,
)

RATE = 16000


def write_pattern_wav(path: Path, seconds: float, *, rate: int = RATE) -> int:
    """WAV mono 16 bits em que cada amostra é única (i % 30000): permite provar que nada se perde/duplica."""
    n = int(seconds * rate)
    data = array.array("h", (i % 30000 for i in range(n)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return n


def read_all_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes())


# ---------------------------------------------------------------------------
# 3. Verificação do FFmpeg
# ---------------------------------------------------------------------------


class TestFFmpeg:
    def test_found(self, ffmpeg):
        assert ffmpeg.version and Path(ffmpeg.ffmpeg).name.lower().startswith("ffmpeg")
        assert Path(ffmpeg.ffprobe).name.lower().startswith("ffprobe")
        ok, msg = ap.check_ffmpeg()
        assert ok and "FFmpeg" in msg

    def test_not_installed(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        monkeypatch.delenv("FFMPEG_DIR", raising=False)
        monkeypatch.delenv("FFMPEG_BIN", raising=False)
        monkeypatch.setattr(ap, "_COMMON_FFMPEG_DIRS_WIN", ())
        with pytest.raises(FFmpegNotFoundError) as exc:
            ap.find_ffmpeg()
        assert "PATH" in exc.value.user_message
        ok, msg = ap.check_ffmpeg()
        assert not ok and "FFmpeg" in msg

    def test_ffprobe_missing_has_its_own_message(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "C:\\x\\ffmpeg.exe" if name == "ffmpeg" else None)
        monkeypatch.delenv("FFMPEG_DIR", raising=False)
        monkeypatch.delenv("FFMPEG_BIN", raising=False)
        monkeypatch.setattr(ap, "_COMMON_FFMPEG_DIRS_WIN", ())
        with pytest.raises(FFmpegNotFoundError, match="FFprobe"):
            ap.find_ffmpeg()


# ---------------------------------------------------------------------------
# Inspeção de mídia
# ---------------------------------------------------------------------------


class TestProbe:
    def test_audio_file(self, make_audio, ffmpeg, tmp_path):
        f = make_audio(tmp_path / "a.wav", [("tone", 3), ("silence", 1), ("tone", 3)])
        info = ap.probe_media(f, ffmpeg)
        assert info.duration == pytest.approx(7.0, abs=0.1)
        assert info.audio_streams == 1 and not info.has_video and info.audio_sample_rate == RATE

    def test_corrupted_file(self, ffmpeg, tmp_path):
        f = tmp_path / "ruim.mp3"
        f.write_bytes(random.Random(1).randbytes(4096))
        with pytest.raises(CorruptedMediaError):
            ap.probe_media(f, ffmpeg)

    def test_video_without_audio(self, ffmpeg, tmp_path):
        f = tmp_path / "mudo.mp4"
        r = subprocess.run([ffmpeg.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                            "color=c=black:s=64x64:d=1:r=10", "-c:v", "mpeg4", str(f)], capture_output=True)
        if r.returncode != 0:
            pytest.skip("este FFmpeg não consegue gerar vídeo de teste")
        with pytest.raises(CorruptedMediaError, match="faixa de áudio"):
            ap.probe_media(f, ffmpeg)


# ---------------------------------------------------------------------------
# 4. Extração de áudio
# ---------------------------------------------------------------------------


class TestExtractAudio:
    def test_converts_to_16khz_mono_and_never_touches_original(self, make_audio, ffmpeg, tmp_path):
        src = make_audio(tmp_path / "origem.wav", [("tone", 3), ("silence", 1)], rate=44100, channels=2)
        before = sha256(src)
        calls = []
        info = ap.extract_audio(src, tmp_path / "out" / "work.wav", ffmpeg=ffmpeg, duration=4.0,
                                progress_cb=lambda f, s: calls.append((f, s)))
        assert (info.sample_rate, info.channels, info.sample_width) == (16000, 1, 2)
        assert info.duration == pytest.approx(4.0, abs=0.1)
        assert sha256(src) == before
        assert calls and calls[-1][0] == 1.0
        fractions = [c[0] for c in calls if c[0] is not None]
        assert fractions == sorted(fractions)  # progresso real e crescente
        assert not list((tmp_path / "out").glob("*.partial.wav"))

    @pytest.mark.parametrize("name,codec", [("a.mp3", "libmp3lame"), ("a.flac", "flac"), ("a.m4a", "aac")])
    def test_other_audio_formats(self, make_audio, ffmpeg, tmp_path, name, codec):
        try:
            src = make_audio(tmp_path / name, [("tone", 2)], rate=44100, channels=2, codec=codec)
        except subprocess.CalledProcessError:
            pytest.skip(f"codec {codec} indisponível neste FFmpeg")
        info = ap.extract_audio(src, tmp_path / "w.wav", ffmpeg=ffmpeg)
        assert info.sample_rate == 16000 and info.duration == pytest.approx(2.0, abs=0.2)

    def test_existing_valid_wav_is_reused(self, make_audio, ffmpeg, tmp_path):
        src = make_audio(tmp_path / "o.wav", [("tone", 2)])
        dest = tmp_path / "w.wav"
        ap.extract_audio(src, dest, ffmpeg=ffmpeg)
        mtime = dest.stat().st_mtime_ns
        calls = []
        ap.extract_audio(src, dest, ffmpeg=ffmpeg, progress_cb=lambda f, s: calls.append(f))
        assert dest.stat().st_mtime_ns == mtime and calls == []

    def test_corrupted_source(self, ffmpeg, tmp_path):
        bad = tmp_path / "ruim.mp3"
        bad.write_bytes(random.Random(2).randbytes(8192))
        with pytest.raises(AudioProcessingError):
            ap.extract_audio(bad, tmp_path / "w.wav", ffmpeg=ffmpeg)
        assert not (tmp_path / "w.wav").exists() and not (tmp_path / "w.partial.wav").exists()

    def test_cancel_leaves_no_partial_file(self, make_audio, ffmpeg, tmp_path):
        src = make_audio(tmp_path / "o.wav", [("tone", 20)])
        with pytest.raises(TaskCancelledError):
            ap.extract_audio(src, tmp_path / "w.wav", ffmpeg=ffmpeg, cancel_check=lambda: True)
        assert not (tmp_path / "w.wav").exists() and not (tmp_path / "w.partial.wav").exists()

    def test_destination_cannot_be_the_original(self, make_audio, ffmpeg, tmp_path):
        src = make_audio(tmp_path / "o.wav", [("tone", 1)])
        with pytest.raises(AudioProcessingError):
            ap.extract_audio(src, src, ffmpeg=ffmpeg)

    def test_not_enough_disk_space(self, make_audio, ffmpeg, tmp_path):
        src = make_audio(tmp_path / "o.wav", [("tone", 1)])
        with pytest.raises(DiskSpaceError):
            ap.extract_audio(src, tmp_path / "w.wav", ffmpeg=ffmpeg, duration=1e12)  # ~ 32 PB

    def test_read_wav_info_rejects_wrong_format(self, tmp_path):
        wrong = tmp_path / "w44.wav"
        write_pattern_wav(wrong, 1, rate=44100)
        with pytest.raises(AudioProcessingError, match="formato"):
            ap.read_wav_info(wrong)
        junk = tmp_path / "junk.wav"
        junk.write_bytes(b"isto nao e um wav")
        with pytest.raises(AudioProcessingError):
            ap.read_wav_info(junk)


# ---------------------------------------------------------------------------
# 5. Detecção de pausas
# ---------------------------------------------------------------------------


class TestSilenceParsing:
    def test_pairs_and_offset(self):
        lines = ["[silencedetect @ 0x1] silence_start: 2.5",
                 "[silencedetect @ 0x1] silence_end: 3.7 | silence_duration: 1.2"]
        res = ap.parse_silencedetect_output(lines, offset=100)
        assert len(res) == 1
        assert res[0].start == pytest.approx(102.5) and res[0].end == pytest.approx(103.7)

    def test_open_silence_is_closed_at_window_end(self):
        res = ap.parse_silencedetect_output(["silence_start: 5.0"], offset=10, window_end=20)
        assert res == [SilenceInterval(15.0, 20)]

    def test_open_silence_ignored_without_window_end(self):
        assert ap.parse_silencedetect_output(["silence_start: 5.0"]) == []

    def test_end_without_start_starts_at_window_beginning(self):
        res = ap.parse_silencedetect_output(["silence_end: 1.0 | silence_duration: 1.0"], offset=50)
        assert res == [SilenceInterval(50, 51.0)]

    def test_negative_start_is_clamped(self):
        res = ap.parse_silencedetect_output(["silence_start: -0.001", "silence_end: 0.8 | silence_duration: 0.8"])
        assert res[0].start == 0.0

    def test_multiple(self):
        lines = ["silence_start: 1", "silence_end: 2 | silence_duration: 1", "silence_start: 5", "silence_end: 6.5 | x"]
        assert [(s.start, s.end) for s in ap.parse_silencedetect_output(lines)] == [(1, 2), (5, 6.5)]

    def test_interval_properties(self):
        s = SilenceInterval(10, 12)
        assert s.duration == 2 and s.midpoint == 11


class TestSilenceSettings:
    def test_filter_string_matches_spec(self):
        assert SilenceSettings().filter_string == "silencedetect=noise=-35dB:d=0.4"
        assert SilenceSettings(noise_db=-42.5, min_duration=0.25).filter_string == "silencedetect=noise=-42.5dB:d=0.25"

    def test_roundtrip_and_defaults(self):
        s = SilenceSettings(-40, 0.5, 30)
        assert SilenceSettings.from_dict(s.to_dict()) == s
        assert SilenceSettings.from_dict(None) == SilenceSettings()


class TestDetectSilences:
    def test_finds_the_pause(self, make_audio, ffmpeg, tmp_path):
        f = make_audio(tmp_path / "a.wav", [("tone", 3), ("silence", 1), ("tone", 3)])
        res = ap.detect_silences(f, 0, 7, ffmpeg=ffmpeg)
        assert len(res) == 1
        assert res[0].start == pytest.approx(3.0, abs=0.15) and res[0].end == pytest.approx(4.0, abs=0.15)

    def test_window_offsets_are_absolute(self, make_audio, ffmpeg, tmp_path):
        f = make_audio(tmp_path / "a.wav", [("tone", 3), ("silence", 1), ("tone", 3)])
        res = ap.detect_silences(f, 2, 6, ffmpeg=ffmpeg)
        assert len(res) == 1 and res[0].start == pytest.approx(3.0, abs=0.15)

    def test_no_pause_in_continuous_tone(self, make_audio, ffmpeg, tmp_path):
        f = make_audio(tmp_path / "a.wav", [("tone", 5)])
        assert ap.detect_silences(f, 0, 5, ffmpeg=ffmpeg) == []

    def test_threshold_is_configurable(self, make_audio, ffmpeg, tmp_path):
        f = make_audio(tmp_path / "a.wav", [("tone", 3), ("silence", 1), ("tone", 3)])
        # com um limite muito alto (-5 dB) o próprio tom (≈ -21 dB) conta como "silêncio"
        loud = ap.detect_silences(f, 0, 7, settings=SilenceSettings(noise_db=-5), ffmpeg=ffmpeg)
        assert loud and loud[0].duration > 5

    def test_empty_window(self, make_audio, ffmpeg, tmp_path):
        f = make_audio(tmp_path / "a.wav", [("tone", 1)])
        assert ap.detect_silences(f, 5, 5, ffmpeg=ffmpeg) == []


# ---------------------------------------------------------------------------
# 6. Planejamento e divisão em blocos
# ---------------------------------------------------------------------------


class TestChooseCutPoint:
    def test_prefers_a_longer_pause_even_if_farther(self):
        res = ap.choose_cut_point([SilenceInterval(10, 10.5), SilenceInterval(20, 22)], 15, 5, 25)
        assert res == pytest.approx(21)

    def test_prefers_closer_when_similar(self):
        res = ap.choose_cut_point([SilenceInterval(14, 14.4), SilenceInterval(30, 32)], 15, 0, 40)
        assert res == pytest.approx(14.2)

    def test_pauses_are_clipped_to_the_window(self):
        assert ap.choose_cut_point([SilenceInterval(0, 3)], 15, 5, 25) is None
        assert ap.choose_cut_point([SilenceInterval(4, 6)], 15, 5, 25) == pytest.approx(5.5)

    def test_none_without_pauses(self):
        assert ap.choose_cut_point([], 15, 5, 25) is None


class TestPlanChunks:
    def test_nominal_cuts_without_finder(self):
        chunks = ap.plan_chunks(1500 * RATE, chunk_seconds=600, search_window=60)
        assert [(c.start, c.end, c.cut_reason) for c in chunks] == [
            (0, 600, "nominal"), (600, 1200, "nominal"), (1200, 1500, "end")]

    def test_short_audio_is_a_single_chunk(self):
        chunks = ap.plan_chunks(605 * RATE, chunk_seconds=600, search_window=60)
        assert len(chunks) == 1 and chunks[0].end_sample == 605 * RATE

    def test_cut_moves_to_the_pause_and_next_limit_follows_the_real_cut(self):
        calls = []

        def finder(lo, hi):
            calls.append((lo, hi))
            return [SilenceInterval(595, 597)] if len(calls) == 1 else []

        chunks = ap.plan_chunks(1500 * RATE, chunk_seconds=600, search_window=60, silence_finder=finder)
        assert calls[0] == (540.0, 660.0)
        assert [c.cut_reason for c in chunks] == ["silence", "nominal", "end"]
        assert chunks[0].end == pytest.approx(596.0)
        assert chunks[1].end == pytest.approx(596.0 + 600)  # limite nominal contado a partir do corte real
        ap.verify_chunk_continuity(chunks, 1500 * RATE)

    def test_finder_without_result_uses_nominal_limit(self):
        chunks = ap.plan_chunks(1500 * RATE, chunk_seconds=600, search_window=60, silence_finder=lambda a, b: [])
        assert chunks[0].end == 600

    def test_continuity_property_for_many_random_durations(self):
        rng = random.Random(1234)
        for _ in range(200):
            total = rng.randint(1, 5 * 3600 * RATE)
            chunk_s = rng.choice([60, 300, 600, 900])
            window_s = rng.choice([0, 10, 30, 60])

            def finder(lo, hi, rng=rng):
                out = []
                for _ in range(rng.randint(0, 3)):
                    a = rng.uniform(lo, hi)
                    out.append(SilenceInterval(a, min(hi + 5, a + rng.uniform(0.4, 3))))
                return out

            chunks = ap.plan_chunks(total, chunk_seconds=chunk_s, search_window=window_s, silence_finder=finder)
            ap.verify_chunk_continuity(chunks, total)  # sem lacunas, sem sobreposição, do 0 ao fim
            assert sum(c.frames for c in chunks) == total
            eff_window = int(round(min(window_s, chunk_s / 2) * RATE))
            nominal = chunk_s * RATE
            for c in chunks[:-1]:
                assert nominal - eff_window <= c.frames <= nominal + eff_window
            assert 0 < chunks[-1].frames <= nominal + eff_window

    def test_cancel(self):
        with pytest.raises(TaskCancelledError):
            ap.plan_chunks(5000 * RATE, cancel_check=lambda: True)

    def test_invalid_arguments(self):
        with pytest.raises(AudioProcessingError):
            ap.plan_chunks(0)
        with pytest.raises(ValueError):
            ap.plan_chunks(1000, chunk_seconds=0)

    def test_progress_callback_ends_complete(self):
        seen = []
        ap.plan_chunks(2500 * RATE, progress_cb=lambda n, t: seen.append((n, t)))
        assert seen[-1][0] == seen[-1][1]


class TestVerifyContinuity:
    def spec(self, i, a, b):
        return ChunkSpec(i, a, b)

    def test_ok(self):
        ap.verify_chunk_continuity([self.spec(0, 0, 100), self.spec(1, 100, 250)], 250)

    @pytest.mark.parametrize("chunks,total", [
        ([], 100),
        ([ChunkSpec(0, 5, 100)], 100),                                   # não começa em 0
        ([ChunkSpec(0, 0, 100), ChunkSpec(1, 110, 200)], 200),           # lacuna
        ([ChunkSpec(0, 0, 100), ChunkSpec(1, 90, 200)], 200),            # sobreposição
        ([ChunkSpec(0, 0, 100), ChunkSpec(1, 100, 150)], 200),           # termina antes do fim
        ([ChunkSpec(0, 0, 0), ChunkSpec(1, 0, 100)], 100),               # bloco vazio
        ([ChunkSpec(0, 0, 100), ChunkSpec(5, 100, 200)], 200),           # índice fora de ordem
    ])
    def test_detects_problems(self, chunks, total):
        with pytest.raises(AudioProcessingError):
            ap.verify_chunk_continuity(chunks, total)

    def test_chunk_spec_serialization_and_times(self):
        c = ChunkSpec(3, 16000 * 600, 16000 * 1180, cut_reason="silence")
        d = c.to_dict()
        assert (d["start"], d["end"], d["duration"]) == (600.0, 1180.0, 580.0)
        again = ChunkSpec.from_dict(d)
        assert again == c and again.frames == 16000 * 580
        assert ap.chunk_offset_seconds(c) == 600.0


class TestSplitWav:
    def test_no_audio_is_lost_or_duplicated(self, tmp_path):
        wav = tmp_path / "work.wav"
        total = write_pattern_wav(wav, 25)
        chunks = ap.plan_chunks(total, chunk_seconds=10, search_window=2)
        assert [(c.start, c.end) for c in chunks] == [(0, 10), (10, 20), (20, 25)]
        specs = ap.split_wav(wav, chunks, tmp_path / "chunks")
        files = [Path(s.path) for s in specs]
        assert [f.name for f in files] == ["chunk_0000.wav", "chunk_0001.wav", "chunk_0002.wav"]
        assert [ap.read_wav_info(f).frames for f in files] == [c.frames for c in chunks]
        # a concatenação dos blocos é IDÊNTICA ao WAV original
        assert b"".join(read_all_frames(f) for f in files) == read_all_frames(wav)

    def test_chunk_boundaries_follow_the_plan(self, tmp_path):
        wav = tmp_path / "work.wav"
        write_pattern_wav(wav, 25)
        chunks = [ChunkSpec(0, 0, 123_457), ChunkSpec(1, 123_457, 400_000)]  # cortes em amostras "estranhas"
        specs = ap.split_wav(wav, chunks, tmp_path / "c")
        original = read_all_frames(wav)
        assert read_all_frames(Path(specs[0].path)) == original[: 123_457 * 2]
        assert read_all_frames(Path(specs[1].path)) == original[123_457 * 2:]

    def test_valid_chunks_are_reused_and_corrupted_ones_rebuilt(self, tmp_path):
        wav = tmp_path / "work.wav"
        total = write_pattern_wav(wav, 25)
        chunks = ap.plan_chunks(total, chunk_seconds=10, search_window=2)
        specs = ap.split_wav(wav, chunks, tmp_path / "c")
        f0, f1 = Path(specs[0].path), Path(specs[1].path)
        m0 = f0.stat().st_mtime_ns
        f1.write_bytes(f1.read_bytes()[:100])  # arquivo truncado (interrupção)
        ap.split_wav(wav, chunks, tmp_path / "c")
        assert f0.stat().st_mtime_ns == m0
        assert ap.read_wav_info(f1).frames == chunks[1].frames

    def test_cancel_leaves_no_partial_chunk(self, tmp_path):
        wav = tmp_path / "work.wav"
        total = write_pattern_wav(wav, 25)
        chunks = ap.plan_chunks(total, chunk_seconds=10, search_window=2)
        counter = {"n": 0}

        def cancel():
            counter["n"] += 1
            return counter["n"] > 3

        with pytest.raises(TaskCancelledError):
            ap.split_wav(wav, chunks, tmp_path / "c", cancel_check=cancel)
        assert not list((tmp_path / "c").glob("*.partial.wav"))

    def test_chunk_past_end_of_wav(self, tmp_path):
        wav = tmp_path / "work.wav"
        write_pattern_wav(wav, 1)
        with pytest.raises(AudioProcessingError):
            ap.export_chunk(wav, ChunkSpec(0, 0, 999_999), tmp_path / "x.wav")

    def test_progress_and_remove_files(self, tmp_path):
        wav = tmp_path / "work.wav"
        total = write_pattern_wav(wav, 25)
        chunks = ap.plan_chunks(total, chunk_seconds=10, search_window=2)
        seen = []
        specs = ap.split_wav(wav, chunks, tmp_path / "c", progress_cb=lambda i, n: seen.append((i, n)))
        assert seen == [(1, 3), (2, 3), (3, 3)]
        ap.remove_chunk_files(specs)
        assert not list((tmp_path / "c").glob("*.wav"))


class TestBuildChunkPlanWithRealSilence:
    def test_cuts_land_in_the_pauses(self, make_audio, ffmpeg, tmp_path):
        wav = make_audio(tmp_path / "a.wav", [("tone", 8), ("silence", 1), ("tone", 8), ("silence", 1), ("tone", 8)])
        info = ap.read_wav_info(wav)
        chunks = ap.build_chunk_plan(wav, chunk_seconds=10, settings=SilenceSettings(search_window=4), ffmpeg=ffmpeg)
        assert [c.cut_reason for c in chunks] == ["silence", "silence", "end"]
        assert chunks[0].end == pytest.approx(8.5, abs=0.15)
        assert chunks[1].end == pytest.approx(17.5, abs=0.15)
        ap.verify_chunk_continuity(chunks, info.frames)

    def test_without_silence_detection_uses_nominal_limits(self, tmp_path):
        wav = tmp_path / "a.wav"
        write_pattern_wav(wav, 26)
        chunks = ap.build_chunk_plan(wav, chunk_seconds=10, settings=SilenceSettings(search_window=4), use_silence=False)
        assert [(c.start, c.end, c.cut_reason) for c in chunks] == [(0, 10, "nominal"), (10, 20, "nominal"), (20, 26, "end")]

    def test_no_pause_in_window_falls_back_to_nominal(self, make_audio, ffmpeg, tmp_path):
        wav = make_audio(tmp_path / "a.wav", [("tone", 26)])
        chunks = ap.build_chunk_plan(wav, chunk_seconds=10, settings=SilenceSettings(search_window=4), ffmpeg=ffmpeg)
        assert [c.cut_reason for c in chunks] == ["nominal", "nominal", "end"]
        assert chunks[0].end == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Arquivo longo (opcional): RUN_SLOW=1
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not env_flag("RUN_SLOW"), reason="defina RUN_SLOW=1 para processar uma hora de áudio sintético")
def test_one_hour_audio_memory_and_continuity(ffmpeg, tmp_path):
    wav = tmp_path / "longo.wav"
    # tom contínuo com 1 s de silêncio ao fim de cada 100 s
    subprocess.run(
        [ffmpeg.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "sine=frequency=300:sample_rate=16000:duration=3600",
         "-af", "volume='if(lt(mod(t,100),99),1,0)':eval=frame", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
        check=True, capture_output=True,
    )
    info = ap.read_wav_info(wav)
    proc = psutil.Process()
    baseline = proc.memory_info().rss
    peak = {"v": baseline}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak["v"] = max(peak["v"], proc.memory_info().rss)
            time.sleep(0.05)

    t = threading.Thread(target=sampler)
    t.start()
    try:
        started = time.monotonic()
        chunks = ap.build_chunk_plan(wav, chunk_seconds=600, settings=SilenceSettings(), ffmpeg=ffmpeg)
        specs = ap.split_wav(wav, chunks, tmp_path / "chunks")
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        t.join()

    ap.verify_chunk_continuity(chunks, info.frames)
    assert 5 <= len(chunks) <= 7
    assert all(Path(s.path).is_file() for s in specs)
    for c in chunks[:-1]:
        assert c.cut_reason == "silence"
        assert (c.end % 100) == pytest.approx(99.5, abs=0.3)  # cortes no meio das pausas
    assert sum(ap.read_wav_info(s.path).frames for s in specs) == info.frames
    delta_mb = (peak["v"] - baseline) / (1024 * 1024)
    print(f"\n[1 h de áudio] {len(chunks)} blocos em {elapsed:.1f} s; aumento de RAM do processo: {delta_mb:.0f} MB")
    assert delta_mb < 200
