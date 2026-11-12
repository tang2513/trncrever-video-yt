"""Testes de src/downloader.py: validação de links, classificação de erros e progresso (sem rede)."""

from __future__ import annotations

import pytest

import src.downloader as dl
from src.utils import (
    AuthenticationRequiredError,
    ContentUnavailableError,
    DownloadError,
    InvalidInputError,
    NetworkError,
    ProtectedContentError,
)

VID = "dQw4w9WgXcQ"


class TestYouTubeUrls:
    @pytest.mark.parametrize("url", [
        f"https://www.youtube.com/watch?v={VID}",
        f"http://youtube.com/watch?v={VID}&t=30s",
        f"https://m.youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}?si=abc",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/live/{VID}",
        f"https://www.youtube.com/embed/{VID}",
        f"www.youtube.com/watch?v={VID}",
        f"youtube.com/watch?v={VID}",
        f'  "https://www.youtube.com/watch?v={VID}"  ',
        f"https://www.youtube.com/watch?v={VID}&list=PL12345678&index=2",
    ])
    def test_valid_links_are_normalized(self, url):
        v = dl.validate_url(url)
        assert v.source_type == dl.SOURCE_YOUTUBE
        assert v.media_id == VID
        assert v.url == f"https://www.youtube.com/watch?v={VID}"

    def test_playlist_is_rejected(self):
        with pytest.raises(InvalidInputError, match="playlist"):
            dl.validate_url("https://www.youtube.com/playlist?list=PL12345678")

    def test_channel_is_rejected(self):
        with pytest.raises(InvalidInputError, match="canal"):
            dl.validate_url("https://www.youtube.com/@algumcanal")

    def test_invalid_video_id(self):
        with pytest.raises(InvalidInputError, match="inválido"):
            dl.validate_url("https://www.youtube.com/watch?v=curto")

    def test_missing_video_id(self):
        with pytest.raises(InvalidInputError):
            dl.validate_url("https://www.youtube.com/watch")


class TestTwitchUrls:
    @pytest.mark.parametrize("url", [
        "https://www.twitch.tv/videos/123456789",
        "https://twitch.tv/videos/123456789?t=1h2m3s",
        "https://m.twitch.tv/videos/123456789",
        "https://www.twitch.tv/canal/v/123456789",
    ])
    def test_valid_vods(self, url):
        v = dl.validate_url(url)
        assert (v.source_type, v.media_id, v.url) == (dl.SOURCE_TWITCH, "123456789", "https://www.twitch.tv/videos/123456789")

    def test_channel_or_live_is_rejected(self):
        with pytest.raises(InvalidInputError, match="VOD"):
            dl.validate_url("https://www.twitch.tv/algumcanal")

    def test_clip_is_rejected(self):
        with pytest.raises(InvalidInputError, match="Clipes"):
            dl.validate_url("https://clips.twitch.tv/AlgumClipe")


class TestGenericValidation:
    @pytest.mark.parametrize("bad", ["", "   ", "https://example.com/video", "ftp://x", "texto qualquer"])
    def test_rejected(self, bad):
        with pytest.raises(InvalidInputError):
            dl.validate_url(bad)

    def test_detect_source_type(self):
        assert dl.detect_source_type(r"C:\Videos\a.mp4") == dl.SOURCE_LOCAL
        assert dl.detect_source_type(f"https://youtu.be/{VID}") == dl.SOURCE_YOUTUBE
        assert dl.detect_source_type("https://www.twitch.tv/videos/1") == dl.SOURCE_TWITCH
        assert dl.detect_source_type("https://example.com/a") == dl.SOURCE_LOCAL

    def test_looks_like_url(self):
        assert dl.looks_like_url("https://x.com") and dl.looks_like_url("www.x.com")
        assert not dl.looks_like_url(r"C:\a.mp4")


class TestErrorClassification:
    @pytest.mark.parametrize("stderr,expected", [
        ("ERROR: [youtube] abc: Video unavailable", ContentUnavailableError),
        ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access", ContentUnavailableError),
        ("ERROR: [twitch:vod] 1: The VOD does not exist", ContentUnavailableError),
        ("ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate", AuthenticationRequiredError),
        ("ERROR: [youtube] abc: Sign in to confirm you’re not a bot. Use --cookies-from-browser", AuthenticationRequiredError),
        ("ERROR: [youtube] abc: This video is DRM protected", ProtectedContentError),
        ("ERROR: [youtube] abc: Join this channel to get access to members-only content", ProtectedContentError),
        ("ERROR: Unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>", NetworkError),
        ("ERROR: unable to download video data: HTTP Error 503: Service Unavailable", NetworkError),
        ("WARNING: ... ERROR: algo muito estranho aconteceu", DownloadError),
    ])
    def test_classification(self, stderr, expected):
        exc = dl.classify_ytdlp_error(stderr)
        assert type(exc) is expected
        assert exc.user_message

    def test_live_stream_has_specific_message(self):
        exc = dl.classify_ytdlp_error("ERROR: [youtube] abc: This live event will begin in 2 hours.")
        assert isinstance(exc, ContentUnavailableError) and "ao vivo" in exc.user_message

    def test_twitch_unavailable_message(self):
        exc = dl.classify_ytdlp_error("ERROR: Video unavailable", source_type=dl.SOURCE_TWITCH)
        assert "VOD" in exc.user_message and "expirado" in exc.user_message

    def test_technical_detail_is_kept_for_the_log(self):
        exc = dl.classify_ytdlp_error("linha 1\nERROR: [youtube] abc: Video unavailable\nlinha 3")
        assert "Video unavailable" in (exc.technical or "")

    def test_network_errors_are_the_only_retryable_ones(self):
        assert not issubclass(ContentUnavailableError, NetworkError)
        assert not issubclass(AuthenticationRequiredError, NetworkError)


class TestProgress:
    def test_bytes(self):
        p = dl.parse_progress_line("[PROG] 1024|2048|NA|NA|NA")
        assert (p.downloaded_bytes, p.total_bytes) == (1024, 2048) and p.fraction == 0.5

    def test_estimate_when_total_unknown(self):
        p = dl.parse_progress_line("[PROG] 1000|NA|4000|NA|NA")
        assert p.fraction == pytest.approx(0.25)

    def test_fragments(self):
        p = dl.parse_progress_line("[PROG] 0|NA|NA|3|10")
        assert p.fraction == pytest.approx(0.3)

    def test_no_information_gives_no_fraction(self):
        assert dl.parse_progress_line("[PROG] 500|NA|NA|NA|NA").fraction is None

    def test_other_lines_are_ignored(self):
        assert dl.parse_progress_line("[download] 10% of 5MiB") is None

    def test_fraction_is_clamped(self):
        assert dl.parse_progress_line("[PROG] 5000|1000|NA|NA|NA").fraction == 1.0

    def test_to_dict(self):
        assert dl.parse_progress_line("[PROG] 1|2|NA|NA|NA").to_dict()["fraction"] == 0.5


class TestEstimates:
    def test_known_size(self):
        assert dl.estimate_download_bytes({"filesize": 1000}) == 1100

    def test_from_bitrate(self):
        assert dl.estimate_download_bytes({"duration": 3600, "abr": 128}) == int(3600 * 128000 / 8 * 1.2)

    def test_unknown_is_conservative(self):
        assert dl.estimate_download_bytes({}) == 2 * 1024 ** 3


class TestCookies:
    def test_copy_keeps_original_untouched(self, tmp_path):
        src = tmp_path / "cookies.txt"
        src.write_text("# Netscape HTTP Cookie File\n.twitch.tv\tTRUE\t/\tTRUE\t0\tname\tvalue\n", encoding="utf-8")
        before = src.read_bytes()
        copy = dl.prepare_cookies_copy(src, tmp_path / "work")
        assert copy != src and copy.read_bytes() == before
        copy.write_text("alterado")
        assert src.read_bytes() == before

    def test_none(self, tmp_path):
        assert dl.prepare_cookies_copy(None, tmp_path) is None

    def test_missing_and_invalid(self, tmp_path):
        with pytest.raises(InvalidInputError):
            dl.prepare_cookies_copy(tmp_path / "nao_existe.txt", tmp_path)
        bad = tmp_path / "bad.txt"
        bad.write_text("isto nao e cookie", encoding="utf-8")
        with pytest.raises(InvalidInputError, match="Netscape"):
            dl.prepare_cookies_copy(bad, tmp_path)
