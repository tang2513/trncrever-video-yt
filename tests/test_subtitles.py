"""Testes de src/subtitles.py: leitura de VTT/SRT, escolha de legendas e geração de TXT/SRT/VTT."""

from __future__ import annotations

import io
from datetime import timedelta

import pytest
import srt

import src.subtitles as sb
from src.subtitles import Segment, TranscriptHeader
from src.utils import SubtitleError


def S(start, end, text):
    return Segment(float(start), float(end), text)


# ---------------------------------------------------------------------------
# 8. Leitura de legendas VTT
# ---------------------------------------------------------------------------

SIMPLE_VTT = """\ufeffWEBVTT
Kind: captions
Language: pt-BR

NOTE isto é um comentário
que ocupa duas linhas

STYLE
::cue { color: red }

1
00:00:01.000 --> 00:00:03.500 align:start
Olá, <i>mundo</i> &amp; amigos

00:00:04.000 --> 00:00:06.000
<v Maria>Primeira linha
segunda linha</v>

00:01:00.250 --> 00:01:02.000
Uma hora depois
"""

# Formato "rolante" das legendas automáticas do YouTube (linha com um único espaço após o tempo).
ROLLING_VTT = (
    "WEBVTT\nKind: captions\nLanguage: pt\n\n"
    "00:00:00.160 --> 00:00:02.230 align:start position:0%\n"
    " \n"
    "olá<00:00:00.640><c> pessoal</c><00:00:01.040><c> tudo</c><00:00:01.439><c> bem</c>\n\n"
    "00:00:02.230 --> 00:00:02.240 align:start position:0%\n"
    "olá pessoal tudo bem\n"
    " \n\n"
    "00:00:02.240 --> 00:00:04.150 align:start position:0%\n"
    "olá pessoal tudo bem\n"
    "hoje<00:00:02.800><c> vamos</c><00:00:03.200><c> falar</c>\n\n"
    "00:00:04.150 --> 00:00:04.160 align:start position:0%\n"
    "hoje vamos falar\n"
    " \n\n"
    "00:00:04.160 --> 00:00:06.000 align:start position:0%\n"
    "hoje vamos falar\n"
    "sobre<00:00:04.500><c> testes</c>\n"
)


class TestParseVtt:
    def test_simple_vtt(self):
        segs = sb.parse_vtt(SIMPLE_VTT)
        assert [s.text for s in segs] == ["Olá, mundo & amigos", "Primeira linha segunda linha", "Uma hora depois"]
        assert (segs[0].start, segs[0].end) == (1.0, 3.5)
        assert segs[2].start == pytest.approx(60.25)

    def test_header_note_and_style_are_ignored(self):
        texts = " ".join(s.text for s in sb.parse_vtt(SIMPLE_VTT))
        assert "comentário" not in texts and "color" not in texts and "WEBVTT" not in texts

    def test_from_file_with_bom(self, tmp_path):
        p = tmp_path / "a.vtt"
        p.write_text(SIMPLE_VTT, encoding="utf-8")
        assert len(sb.parse_vtt(p)) == 3
        assert len(sb.parse_subtitle_file(p)) == 3

    def test_crlf_line_endings(self):
        assert len(sb.parse_vtt(SIMPLE_VTT.replace("\n", "\r\n"))) == 3

    def test_youtube_rolling_captions_are_deduplicated_structurally(self):
        segs = sb.parse_vtt(ROLLING_VTT)
        assert [s.text for s in segs] == ["olá pessoal tudo bem", "hoje vamos falar", "sobre testes"]

    def test_rolling_times_do_not_overlap(self):
        segs = sb.parse_vtt(ROLLING_VTT)
        assert segs[0].start == pytest.approx(0.16)
        for a, b in zip(segs, segs[1:]):
            assert a.end <= b.start + 1e-9

    def test_speaker_repetitions_are_preserved(self):
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nsim\n\n"
               "00:00:02.000 --> 00:00:03.000\nsim\n\n00:00:03.000 --> 00:00:04.000\nsim\n")
        assert [s.text for s in sb.parse_vtt(vtt)] == ["sim", "sim", "sim"]

    def test_empty_and_invalid_input(self):
        assert sb.parse_vtt("WEBVTT\n\n") == []
        assert sb.parse_vtt("isto não é uma legenda\n") == []

    def test_cue_without_text_is_skipped(self):
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<c></c>\n\n00:00:03.000 --> 00:00:04.000\nok\n"
        assert [s.text for s in sb.parse_vtt(vtt)] == ["ok"]

    def test_accepts_file_object(self):
        assert len(sb.parse_vtt(io.StringIO(SIMPLE_VTT))) == 3

    def test_unsupported_extension(self, tmp_path):
        with pytest.raises(SubtitleError):
            sb.parse_subtitle_file(tmp_path / "x.ass")

    def test_clean_caption_text(self):
        assert sb.clean_caption_text("<c.colorE5E5E5>a</c> &lt;b&gt;\u200b  c") == "a <b> c"


class TestParseSrt:
    def test_basic(self):
        content = "1\n00:00:01,000 --> 00:00:02,500\nOlá\n\n2\n00:01:00,000 --> 00:01:01,000\n<i>Tchau</i>\nmundo\n"
        segs = sb.parse_srt(content)
        assert [(s.start, s.end, s.text) for s in segs] == [(1.0, 2.5, "Olá"), (60.0, 61.0, "Tchau mundo")]

    def test_malformed(self):
        with pytest.raises(SubtitleError):
            sb.parse_srt("1\nisto não é tempo\ntexto\n")


# ---------------------------------------------------------------------------
# Escolha da faixa de legenda
# ---------------------------------------------------------------------------


def track(ext="vtt"):
    return [{"ext": ext, "name": "x"}]


class TestChooseSubtitleTrack:
    def test_prefers_creator_subtitles_in_ptbr(self):
        info = {"language": "pt", "subtitles": {"en": track(), "pt-BR": track(), "pt": track()},
                "automatic_captions": {"pt-orig": track(), "pt": track()}}
        sel = sb.choose_subtitle_track(info)
        assert (sel.track.lang, sel.track.kind) == ("pt-BR", "manual") and sel.portuguese_available

    def test_portuguese_order_ptbr_then_pt_then_ptpt(self):
        assert sb.choose_subtitle_track({"subtitles": {"pt-PT": track(), "pt": track()}}).track.lang == "pt"
        assert sb.choose_subtitle_track({"subtitles": {"pt-PT": track()}}).track.lang == "pt-PT"

    def test_falls_back_to_original_automatic_portuguese(self):
        info = {"language": "pt", "automatic_captions": {"pt-orig": track(), "pt": track(), "en": track()}}
        sel = sb.choose_subtitle_track(info)
        assert (sel.track.lang, sel.track.kind) == ("pt-orig", "auto")

    def test_automatic_translations_are_not_accepted(self):
        info = {"language": "en", "automatic_captions": {"en": track(), "pt": track(), "es": track()}}
        sel = sb.choose_subtitle_track(info)
        assert sel.track is None and not sel.portuguese_available
        assert "não possui legendas em português" in sel.message

    def test_no_subtitles_at_all(self):
        sel = sb.choose_subtitle_track({})
        assert sel.track is None and "português" in sel.message

    def test_other_language_only_when_allowed(self):
        info = {"language": "en", "subtitles": {"en": track(), "fr": track()}}
        assert sb.choose_subtitle_track(info, "prefer_pt").track is None
        sel = sb.choose_subtitle_track(info, "any_language")
        assert sel.track.lang == "en" and "não possui legendas em português" in sel.message

    def test_other_language_uses_original_automatic_as_last_resort(self):
        info = {"language": "en", "automatic_captions": {"en": track(), "en-orig": track(), "de": track()}}
        sel = sb.choose_subtitle_track(info, "any_language")
        assert sel.track.lang == "en-orig"

    def test_ignore_mode_never_uses_subtitles(self):
        sel = sb.choose_subtitle_track({"subtitles": {"pt": track()}}, "ignore")
        assert sel.track is None and sel.portuguese_available

    def test_live_chat_is_not_a_subtitle(self):
        assert sb.choose_subtitle_track({"subtitles": {"live_chat": track("json")}}, "any_language").track is None

    def test_track_description_and_roundtrip(self):
        t = sb.choose_subtitle_track({"subtitles": {"pt-BR": track()}}).track
        assert "adicionada pelo criador" in t.description and "Português" in t.description
        assert sb.SubtitleTrack.from_dict(t.to_dict()) == t


# ---------------------------------------------------------------------------
# Parágrafos e marcações de tempo
# ---------------------------------------------------------------------------


class TestParagraphs:
    def test_marker_every_5_minutes_with_real_times(self):
        segs = [S(t, t + 30, f"t{t}") for t in range(0, 720, 30)]
        paras = list(sb.build_paragraphs(segs, "5min"))
        assert [p.start for p in paras] == [0, 300, 600]
        assert all(p.show_timestamp for p in paras)

    def test_marker_every_10_and_15_minutes(self):
        segs = [S(t, t + 30, f"t{t}") for t in range(0, 1800, 30)]
        assert [p.start for p in sb.build_paragraphs(segs, "10min")] == [0, 600, 1200]
        assert [p.start for p in sb.build_paragraphs(segs, "15min")] == [0, 900]

    def test_marker_shows_actual_start_not_rounded(self):
        segs = [S(0, 10, "a"), S(310, 320, "b")]
        assert [p.start for p in sb.build_paragraphs(segs, "5min")] == [0, 310]

    def test_long_silence_does_not_create_empty_sections(self):
        segs = [S(0, 10, "a"), S(2000, 2010, "b")]
        paras = list(sb.build_paragraphs(segs, "5min"))
        assert [p.start for p in paras] == [0, 2000]

    def test_segment_mode_one_line_per_segment(self):
        segs = [S(1, 2, "a"), S(3, 4, "b"), S(5, 6, "   ")]
        paras = list(sb.build_paragraphs(segs, "segment"))
        assert [(p.start, p.text, p.show_timestamp) for p in paras] == [(1, "a", True), (3, "b", True)]

    def test_big_paragraphs_are_split_without_timestamp(self):
        text = "x" * 399 + "."
        segs = [S(i * 10, i * 10 + 9, text) for i in range(6)]
        paras = list(sb.build_paragraphs(segs, "5min"))
        assert len(paras) == 2
        assert paras[0].show_timestamp and not paras[1].show_timestamp
        assert paras[0].text.count("x") + paras[1].text.count("x") == 6 * 399  # nada se perde

    def test_pause_creates_new_paragraph_after_enough_text(self):
        segs = [S(0, 5, "a" * 250), S(20, 25, "b" * 10)]
        paras = list(sb.build_paragraphs(segs, "5min", gap_break=2.5, min_chars_for_gap_break=200))
        assert len(paras) == 2 and not paras[1].show_timestamp

    def test_text_is_never_changed_or_dropped(self):
        words = [f"palavra{i}" for i in range(500)]
        segs = [S(i * 3, i * 3 + 2, w) for i, w in enumerate(words)]
        joined = " ".join(p.text for p in sb.build_paragraphs(segs, "5min"))
        assert joined.split() == words

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            list(sb.build_paragraphs([S(0, 1, "a")], "2min"))


# ---------------------------------------------------------------------------
# 9. Geração de TXT, SRT e VTT
# ---------------------------------------------------------------------------


def header(**kw):
    base = dict(title="Palestra: ação & reação", source_type="Arquivo local", source=r"C:\Videos\palestra.mp4",
                method="Transcrição automática do áudio (faster-whisper)", processed_at="12/11/2026 10:00:00",
                language="pt", language_probability=0.98, model="small", device="CPU · int8", duration=3725.0)
    base.update(kw)
    return TranscriptHeader(**base)


class TestWriteTxt:
    def test_header_and_structure(self, tmp_path):
        segs = [S(0, 4, "Início da fala."), S(5, 9, "Continuação."), S(305, 310, "Depois de cinco minutos.")]
        out = tmp_path / "t.txt"
        n = sb.write_txt(segs, out, header(), "5min")
        text = out.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0] == "TRANSCRIÇÃO"
        for expected in ("Título: Palestra: ação & reação", "Origem: Arquivo local", r"Fonte: C:\Videos\palestra.mp4",
                         "Data de processamento: 12/11/2026 10:00:00", "Idioma: Português (confiança 98%)",
                         "Modelo: small", "Dispositivo: CPU · int8", "Duração: 1h 02min 05s (01:02:05)"):
            assert expected in text
        assert "[00:00:00]\nInício da fala. Continuação.\n" in text
        assert "[00:05:05]\nDepois de cinco minutos.\n" in text
        assert n == 2

    def test_timestamps_are_video_time_not_processing_time(self, tmp_path):
        out = tmp_path / "t.txt"
        sb.write_txt([S(3600, 3605, "uma hora")], out, header(), "5min")
        assert "[01:00:00]" in out.read_text(encoding="utf-8")

    def test_segment_mode(self, tmp_path):
        out = tmp_path / "t.txt"
        sb.write_txt([S(1, 2, "a"), S(65, 70, "b")], out, header(), "segment")
        text = out.read_text(encoding="utf-8")
        assert "[00:00:01] a\n" in text and "[00:01:05] b\n" in text

    def test_no_speech(self, tmp_path):
        out = tmp_path / "t.txt"
        assert sb.write_txt([], out, header(), "5min") == 0
        assert "Nenhuma fala foi detectada" in out.read_text(encoding="utf-8")

    def test_accepts_generators_and_is_utf8_with_lf(self, tmp_path):
        out = tmp_path / "t.txt"
        sb.write_txt((S(i, i + 1, "çãõ") for i in range(3)), out, header(), "5min")
        raw = out.read_bytes()
        assert b"\r\n" not in raw and "çãõ".encode("utf-8") in raw

    def test_no_temp_file_left(self, tmp_path):
        sb.write_txt([S(0, 1, "a")], tmp_path / "t.txt", header(), "5min")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["t.txt"]

    def test_header_without_language(self):
        text = "\n".join(header(language=None, model=None, device=None, duration=None).lines())
        assert "Idioma: Não identificado" in text and "Modelo" not in text


class TestWriteSrtVtt:
    SEGS = [S(0.0, 2.5, "Olá"), S(3661.5, 3665.0, "Linha 1\nLinha 2"), S(4000, 4000, "sem duração"), S(5000, 5001, "  ")]

    def test_srt_roundtrip_with_standard_library(self, tmp_path):
        out = tmp_path / "a.srt"
        assert sb.write_srt(self.SEGS, out) == 3
        subs = list(srt.parse(out.read_text(encoding="utf-8")))
        assert [s.index for s in subs] == [1, 2, 3]
        assert subs[0].start == timedelta(0) and subs[0].end == timedelta(seconds=2.5)
        assert subs[1].start == timedelta(seconds=3661.5)
        assert subs[2].end > subs[2].start  # duração mínima garantida
        assert "00:00:00,000 --> 00:00:02,500" in out.read_text(encoding="utf-8")

    def test_srt_parses_back_with_our_parser(self, tmp_path):
        out = tmp_path / "a.srt"
        sb.write_srt(self.SEGS, out)
        back = sb.parse_srt(out)
        assert back[0].text == "Olá" and back[1].start == pytest.approx(3661.5)

    def test_vtt_format(self, tmp_path):
        out = tmp_path / "a.vtt"
        assert sb.write_vtt(self.SEGS, out) == 3
        text = out.read_text(encoding="utf-8")
        assert text.startswith("WEBVTT\n\n")
        assert "00:00:00.000 --> 00:00:02.500" in text and "01:01:01.500 --> 01:01:05.000" in text
        assert "," not in text.split("-->")[1].split("\n")[0]

    def test_vtt_roundtrip(self, tmp_path):
        out = tmp_path / "a.vtt"
        sb.write_vtt(self.SEGS, out)
        back = sb.parse_vtt(out)
        assert [s.text for s in back] == ["Olá", "Linha 1 Linha 2", "sem duração"]

    def test_arrow_in_text_does_not_break_vtt(self, tmp_path):
        out = tmp_path / "a.vtt"
        sb.write_vtt([S(0, 1, "a --> b")], out)
        assert len(sb.parse_vtt(out)) == 1

    def test_negative_start_is_clamped(self, tmp_path):
        out = tmp_path / "a.srt"
        sb.write_srt([S(-1.0, 1.0, "x")], out)
        assert "00:00:00,000 --> 00:00:01,000" in out.read_text(encoding="utf-8")


class TestSegmentsStorage:
    def test_json_roundtrip(self, tmp_path):
        p = tmp_path / "s.json"
        segs = [S(1.23456, 2.34567, "ação"), S(3, 4, "b")]
        sb.save_segments_json(p, segs, meta={"index": 0})
        back = sb.load_segments_json(p)
        assert [s.text for s in back] == ["ação", "b"] and back[0].start == pytest.approx(1.235)

    def test_iter_segments_from_files_keeps_order_and_is_lazy(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"s{i}.json"
            sb.save_segments_json(p, [S(i * 10, i * 10 + 1, f"bloco{i}")])
            paths.append(p)
        gen = sb.iter_segments_from_files(paths)
        assert iter(gen) is gen  # gerador: lê um bloco por vez
        assert [s.text for s in gen] == ["bloco0", "bloco1", "bloco2"]

    def test_shift(self):
        shifted = sb.shift_segments([S(1, 2, "a")], 600)
        assert (shifted[0].start, shifted[0].end) == (601, 602)

    def test_export_all_creates_three_files_reading_segments_each_time(self, tmp_path):
        calls = []

        def factory():
            calls.append(1)
            return iter([S(0, 1, "a"), S(2, 3, "b")])

        files = sb.export_all(factory, tmp_path / "out", "video", header(), "5min")
        assert len(calls) == 3
        assert {k: v.rsplit(".", 1)[-1] for k, v in files.items()} == {"txt": "txt", "srt": "srt", "vtt": "vtt"}
        assert all((tmp_path / "out" / f"video.{e}").is_file() for e in ("txt", "srt", "vtt"))


def test_language_helpers():
    assert sb.base_language("pt-BR") == "pt" and sb.base_language("zh_Hans") == "zh" and sb.base_language("en-orig") == "en"
    assert sb.language_display("pt-BR") == "Português"
    assert sb.language_display(None) == "Desconhecido"


def test_default_is_one_timestamp_per_speech_segment():
    from src.utils import DEFAULT_TIMESTAMP_MODE, TIMESTAMP_MODES

    assert DEFAULT_TIMESTAMP_MODE == "segment" and {"segment", "30s", "1min"} <= set(TIMESTAMP_MODES)
    segs = [S(0.22, 2, "a"), S(2.28, 4, "b"), S(4.38, 6, "c")]
    assert [(p.start, p.text) for p in sb.build_paragraphs(segs)] == [(0.22, "a"), (2.28, "b"), (4.38, "c")]


def test_30_second_and_1_minute_markers():
    segs = [S(t, t + 5, f"t{t}") for t in range(0, 130, 5)]
    assert [p.start for p in sb.build_paragraphs(segs, "30s")] == [0, 30, 60, 90, 120]
    assert [p.start for p in sb.build_paragraphs(segs, "1min")] == [0, 60, 120]
