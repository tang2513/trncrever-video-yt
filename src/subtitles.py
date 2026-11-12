"""Segmentos, legendas e formatos de saída.

Responsabilidades:

* ``Segment``: unidade comum de texto com início/fim em segundos do vídeo
  ORIGINAL — usada tanto pelas legendas do YouTube quanto pelo faster-whisper.
* Leitura de legendas WebVTT (incluindo o formato "rolante" das legendas
  automáticas do YouTube) e SRT.
* Escolha da melhor faixa de legenda a partir dos metadados do ``yt-dlp``:
  prioridade para pt-BR → pt → pt-PT; legendas automáticas só são aceitas no
  idioma ORIGINAL do vídeo (traduções automáticas não são transcrições fiéis).
* Organização do texto em parágrafos com marcações de tempo configuráveis.
* Gravação de TXT, SRT e VTT em UTF-8, de forma incremental (sem montar o
  resultado inteiro na memória), e leitura/gravação dos segmentos de cada
  bloco em JSON.

Regra de fidelidade: este módulo NUNCA resume, corrige ou remove falas. A única
remoção feita é estrutural — as linhas que o formato rolante do YouTube repete
de um quadro para o outro (artefato do formato, não repetição do palestrante).
"""

from __future__ import annotations

import html
import os
import time
import io
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, TextIO

import srt

from .utils import (
    LANGUAGE_NAMES,
    SubtitleError,
    atomic_write_json,
    format_duration,
    format_timestamp,
    parse_timestamp,
    read_json,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Modelo de dados
# =============================================================================


@dataclass
class Segment:
    """Trecho de fala com tempos absolutos (segundos desde o início do vídeo)."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(start=float(data["start"]), end=float(data["end"]), text=str(data.get("text", "")))

    def shifted(self, offset: float) -> "Segment":
        """Cópia com tempos deslocados (bloco → tempo original do vídeo)."""
        return Segment(self.start + offset, self.end + offset, self.text)


def shift_segments(segments: Iterable[Segment], offset: float) -> list[Segment]:
    return [s.shifted(offset) for s in segments]


def save_segments_json(path: Path | str, segments: Iterable[Segment], meta: Optional[dict[str, Any]] = None) -> None:
    """Grava (atomicamente) os segmentos de um bloco: ``{"meta": {...}, "segments": [...]}``."""
    atomic_write_json(
        path,
        {"meta": meta or {}, "segments": [s.to_dict() for s in segments]},
        keep_backup=False,
    )


def load_segments_json(path: Path | str) -> list[Segment]:
    data = read_json(path, use_backup=False)
    return [Segment.from_dict(d) for d in data.get("segments", [])]


def iter_segments_from_files(paths: Iterable[Path | str]) -> Iterator[Segment]:
    """Percorre os arquivos de bloco em ordem, carregando um bloco por vez (pouca RAM)."""
    for p in paths:
        yield from load_segments_json(p)


# =============================================================================
# Leitura de VTT / SRT
# =============================================================================

_TAG_RE = re.compile(r"<[^>]+>")
_INLINE_TS_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}[.,]\d{3}>|<\d{2}:\d{2}[.,]\d{3}>")
_CUE_TIME_RE = re.compile(
    r"^\s*((?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*((?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})(?:\s+.*)?$"
)


def clean_caption_text(text: str) -> str:
    """Remove tags (``<c>``, ``<i>``, ``<v Nome>``, timestamps inline) e entidades HTML.

    Não altera palavras, pontuação nem a ordem do texto.
    """
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = text.replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


@dataclass
class _RawCue:
    start: float
    end: float
    lines: list[str]

    @property
    def has_inline_timing(self) -> bool:
        return any(_INLINE_TS_RE.search(l) for l in self.lines)


def _iter_raw_vtt_cues(handle: TextIO) -> Iterator[_RawCue]:
    """Lê os cues de um WebVTT (ignora cabeçalho, NOTE, STYLE e REGION)."""
    block: list[str] = []

    def flush(lines: list[str]) -> Optional[_RawCue]:
        if not lines:
            return None
        first = lines[0].strip()
        if first.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            return None
        # Identificador opcional antes da linha de tempo.
        idx = 0
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            idx = 1
        match = _CUE_TIME_RE.match(lines[idx])
        if not match:
            return None
        try:
            start = parse_timestamp(match.group(1))
            end = parse_timestamp(match.group(2))
        except ValueError:
            return None
        return _RawCue(start, end, lines[idx + 1:])

    def waiting_for_text(lines: list[str]) -> bool:
        """O bloco tem só a linha de tempo (e talvez um identificador), sem texto ainda?"""
        return bool(lines) and "-->" in lines[-1]

    for raw in handle:
        line = raw.rstrip("\r\n").lstrip("﻿")
        if line == "" or (line.strip() == "" and not waiting_for_text(block)):
            cue = flush(block)
            if cue:
                yield cue
            block = []
        elif line.strip() == "":
            # O YouTube coloca uma linha com um espaço logo após a linha de tempo para
            # impedir que ela seja vista como "linha em branco" (separador de cues).
            continue
        else:
            block.append(line)
    cue = flush(block)
    if cue:
        yield cue


def _open_text(source: Path | str | TextIO) -> tuple[TextIO, bool]:
    if isinstance(source, (str, Path)) and not (isinstance(source, str) and "\n" in source):
        return open(source, "r", encoding="utf-8-sig", errors="replace"), True
    if isinstance(source, str):
        return io.StringIO(source), True
    return source, False


def parse_vtt(source: Path | str | TextIO) -> list[Segment]:
    """Converte um WebVTT em segmentos.

    * VTT comum (legendas do criador): um segmento por cue, linhas unidas por espaço.
    * VTT "rolante" (legendas automáticas do YouTube, com timestamps inline):
      descarta os cues de ~10 ms que apenas mantêm o texto anterior na tela e a
      primeira linha de cada cue quando ela é a cópia da linha já emitida.
    """
    handle, should_close = _open_text(source)
    try:
        cues = list(_iter_raw_vtt_cues(handle))
    except OSError as exc:
        raise SubtitleError("Não foi possível ler o arquivo de legendas.", technical=repr(exc)) from exc
    finally:
        if should_close:
            handle.close()

    if not cues:
        return []

    rolling = any(c.has_inline_timing for c in cues)
    segments: list[Segment] = []

    if not rolling:
        for cue in cues:
            text = " ".join(t for t in (clean_caption_text(l) for l in cue.lines) if t)
            if text:
                segments.append(Segment(cue.start, max(cue.end, cue.start), text))
        return segments

    last_emitted: Optional[str] = None
    for cue in cues:
        if cue.end - cue.start < 0.05:
            continue  # cue de "retenção" (texto anterior parado na tela)
        lines = [clean_caption_text(l) for l in cue.lines]
        lines = [l for l in lines if l]
        if not lines:
            continue
        if last_emitted is not None and lines[0] == last_emitted:
            if len(lines) > 1:
                lines = lines[1:]  # 1ª linha = cópia estrutural da linha anterior
            elif not cue.has_inline_timing:
                continue  # cue que apenas repete a linha anterior
        for text in lines:
            segments.append(Segment(cue.start, cue.end, text))
            last_emitted = text

    # No formato rolante os cues se sobrepõem; limita o fim ao início do próximo.
    for i in range(len(segments) - 1):
        nxt = segments[i + 1].start
        if segments[i].end > nxt >= segments[i].start:
            segments[i].end = nxt
    return segments


def parse_srt(source: Path | str | TextIO) -> list[Segment]:
    handle, should_close = _open_text(source)
    try:
        content = handle.read()
    finally:
        if should_close:
            handle.close()
    try:
        subs = list(srt.parse(content))
    except srt.SRTParseError as exc:
        raise SubtitleError("O arquivo SRT está malformado.", technical=repr(exc)) from exc
    out: list[Segment] = []
    for sub in subs:
        text = " ".join(t for t in (clean_caption_text(l) for l in sub.content.splitlines()) if t)
        if text:
            out.append(Segment(sub.start.total_seconds(), sub.end.total_seconds(), text))
    return out


def parse_subtitle_file(path: Path | str) -> list[Segment]:
    suffix = Path(path).suffix.lower()
    if suffix == ".vtt":
        return parse_vtt(Path(path))
    if suffix == ".srt":
        return parse_srt(Path(path))
    raise SubtitleError(f"Formato de legenda não suportado: {suffix}")


# =============================================================================
# Escolha de faixa de legenda (metadados do yt-dlp)
# =============================================================================

SUBTITLE_MODES: dict[str, str] = {
    "prefer_pt": "Priorizar legendas em português",
    "any_language": "Usar legendas disponíveis em outros idiomas",
    "ignore": "Ignorar legendas e transcrever o áudio",
}
DEFAULT_SUBTITLE_MODE = "prefer_pt"

#: Ordem de preferência para português.
_PT_PRIORITY = ("pt-br", "pt", "pt-pt")


@dataclass
class SubtitleTrack:
    lang: str            # chave exata usada pelo yt-dlp (ex.: "pt-BR", "pt-orig")
    kind: str            # "manual" (criador) ou "auto" (gerada automaticamente)
    name: str = ""
    exts: list[str] = field(default_factory=list)
    is_original: bool = True  # False para traduções automáticas

    @property
    def base_lang(self) -> str:
        return base_language(self.lang)

    @property
    def description(self) -> str:
        tipo = "adicionada pelo criador" if self.kind == "manual" else "gerada automaticamente"
        return f"{language_display(self.base_lang)} ({self.lang}) — {tipo}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtitleTrack":
        return cls(**data)


@dataclass
class SubtitleSelection:
    track: Optional[SubtitleTrack]
    message: str
    portuguese_available: bool
    available: list[SubtitleTrack] = field(default_factory=list)


def base_language(code: str) -> str:
    """``pt-BR`` → ``pt``; ``en-orig`` → ``en``; ``zh-Hans`` → ``zh``."""
    return (code or "").split("-")[0].split("_")[0].lower()


def language_display(code: Optional[str]) -> str:
    if not code:
        return "Desconhecido"
    return LANGUAGE_NAMES.get(base_language(code), code)


def list_subtitle_tracks(info: dict[str, Any]) -> list[SubtitleTrack]:
    """Lista as faixas de legenda presentes nos metadados do yt-dlp (``-J``)."""
    tracks: list[SubtitleTrack] = []
    for lang, formats in (info.get("subtitles") or {}).items():
        if lang.lower() == "live_chat":
            continue  # chat ao vivo não é legenda
        exts = [f.get("ext", "") for f in formats or [] if isinstance(f, dict)]
        name = next((f.get("name", "") for f in formats or [] if isinstance(f, dict) and f.get("name")), "")
        tracks.append(SubtitleTrack(lang=lang, kind="manual", name=name, exts=exts, is_original=True))

    auto = info.get("automatic_captions") or {}
    orig_bases = {base_language(k) for k in auto if k.lower().endswith("-orig")}
    video_lang = base_language(info.get("language") or "")
    for lang, formats in auto.items():
        exts = [f.get("ext", "") for f in formats or [] if isinstance(f, dict)]
        name = next((f.get("name", "") for f in formats or [] if isinstance(f, dict) and f.get("name")), "")
        b = base_language(lang)
        is_original = lang.lower().endswith("-orig") or (b in orig_bases) or (bool(video_lang) and b == video_lang)
        tracks.append(SubtitleTrack(lang=lang, kind="auto", name=name, exts=exts, is_original=is_original))
    return tracks


def _pt_rank(track: SubtitleTrack) -> int:
    key = track.lang.lower().replace("-orig", "")
    return _PT_PRIORITY.index(key) if key in _PT_PRIORITY else len(_PT_PRIORITY)


def _best_auto_original(tracks: list[SubtitleTrack], base: Optional[str] = None) -> Optional[SubtitleTrack]:
    """Entre as automáticas originais, prefere a chave ``-orig`` (texto sem tradução)."""
    cands = [t for t in tracks if t.kind == "auto" and t.is_original and (base is None or t.base_lang == base)]
    if not cands:
        return None
    cands.sort(key=lambda t: (0 if t.lang.lower().endswith("-orig") else 1, _pt_rank(t), t.lang))
    return cands[0]


def choose_subtitle_track(info: dict[str, Any], mode: str = DEFAULT_SUBTITLE_MODE) -> SubtitleSelection:
    """Escolhe a legenda a usar ou indica que o áudio deve ser transcrito.

    Modos:
      * ``prefer_pt``    → somente português (criador primeiro, depois automática original).
      * ``any_language`` → português primeiro; senão outro idioma (criador, depois automática original).
      * ``ignore``       → nunca usa legendas.
    """
    tracks = list_subtitle_tracks(info)
    manual = [t for t in tracks if t.kind == "manual"]
    pt_manual = sorted((t for t in manual if t.base_lang == "pt"), key=_pt_rank)
    pt_auto = _best_auto_original(tracks, "pt")
    pt_available = bool(pt_manual or pt_auto)

    if mode == "ignore":
        return SubtitleSelection(None, "Legendas ignoradas por escolha do usuário: o áudio será transcrito.",
                                 pt_available, tracks)

    if pt_manual:
        t = pt_manual[0]
        return SubtitleSelection(t, f"Legenda encontrada: {t.description}.", True, tracks)
    if pt_auto:
        return SubtitleSelection(pt_auto, f"Legenda encontrada: {pt_auto.description}.", True, tracks)

    no_pt = "Este vídeo não possui legendas em português."
    if mode != "any_language":
        return SubtitleSelection(None, f"{no_pt} O áudio será baixado e transcrito.", False, tracks)

    video_lang = base_language(info.get("language") or "")
    if manual:
        manual_sorted = sorted(
            manual,
            key=lambda t: (0 if video_lang and t.base_lang == video_lang else 1, 0 if t.base_lang == "en" else 1, t.lang),
        )
        t = manual_sorted[0]
        return SubtitleSelection(t, f"{no_pt} Usando: {t.description}.", False, tracks)
    other_auto = _best_auto_original(tracks, video_lang or None) or _best_auto_original(tracks)
    if other_auto:
        return SubtitleSelection(other_auto, f"{no_pt} Usando: {other_auto.description}.", False, tracks)
    return SubtitleSelection(None, "Este vídeo não possui legendas utilizáveis. O áudio será transcrito.", False, tracks)


# =============================================================================
# Organização do texto (parágrafos + marcações de tempo)
# =============================================================================

TIMESTAMP_INTERVALS: dict[str, Optional[int]] = {
    "30s": 30, "1min": 60, "5min": 300, "10min": 600, "15min": 900, "segment": None,
}

_SENTENCE_END_RE = re.compile(r"[.!?…:;][\"'”»)\]]*$")


@dataclass
class Paragraph:
    start: float
    text: str
    show_timestamp: bool


@dataclass
class TranscriptHeader:
    """Metadados impressos no topo do TXT."""

    title: str
    source_type: str          # "Arquivo local", "YouTube", "Twitch"
    source: str               # caminho ou URL
    method: str               # "Transcrição automática (faster-whisper)" / "Legenda do YouTube ..."
    processed_at: str
    language: Optional[str] = None
    language_probability: Optional[float] = None
    model: Optional[str] = None
    device: Optional[str] = None
    duration: Optional[float] = None

    def lines(self) -> list[str]:
        lang = language_display(self.language) if self.language else "Não identificado"
        if self.language and self.language_probability is not None:
            lang += f" (confiança {self.language_probability:.0%})"
        rows = [
            ("Título", self.title),
            ("Origem", self.source_type),
            ("Fonte", self.source),
            ("Método", self.method),
            ("Data de processamento", self.processed_at),
            ("Idioma", lang),
        ]
        if self.model:
            rows.append(("Modelo", self.model))
        if self.device:
            rows.append(("Dispositivo", self.device))
        if self.duration:
            rows.append(("Duração", f"{format_duration(self.duration)} ({format_timestamp(self.duration)})"))
        return [f"{k}: {v}" for k, v in rows]


def build_paragraphs(
    segments: Iterable[Segment],
    timestamp_mode: str = "segment",
    *,
    gap_break: float = 2.5,
    soft_max_chars: int = 900,
    hard_max_chars: int = 1600,
    min_chars_for_gap_break: int = 200,
) -> Iterator[Paragraph]:
    """Agrupa segmentos em parágrafos legíveis, preservando a ordem e o texto.

    * Modo intervalo (5/10/15 min): uma marcação de tempo sempre que a fala
      cruza o próximo múltiplo do intervalo. A marcação mostra o tempo REAL do
      primeiro segmento da seção (nunca um valor arredondado para frente).
    * Modo ``segment``: cada segmento em sua própria linha com timestamp.
    * Novos parágrafos (sem timestamp) em pausas longas ou quando o parágrafo
      fica grande, sempre em limite de segmento.
    """
    if timestamp_mode not in TIMESTAMP_INTERVALS:
        raise ValueError(f"Modo de marcação inválido: {timestamp_mode}")
    interval = TIMESTAMP_INTERVALS[timestamp_mode]

    if interval is None:
        for seg in segments:
            text = seg.text.strip()
            if text:
                yield Paragraph(seg.start, text, True)
        return

    parts: list[str] = []
    length = 0
    para_start = 0.0
    para_ts = False
    next_marker: Optional[float] = None
    prev_end: Optional[float] = None

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        new_marker = next_marker is None or seg.start >= next_marker
        new_para = False
        if parts and not new_marker:
            gap = seg.start - (prev_end if prev_end is not None else seg.start)
            last = parts[-1]
            if length >= hard_max_chars:
                new_para = True
            elif length >= soft_max_chars and _SENTENCE_END_RE.search(last):
                new_para = True
            elif gap >= gap_break and length >= min_chars_for_gap_break:
                new_para = True
        if parts and (new_marker or new_para):
            yield Paragraph(para_start, " ".join(parts), para_ts)
            parts, length = [], 0
        if not parts:
            para_start = seg.start
            para_ts = new_marker
        if new_marker:
            next_marker = (int(seg.start // interval) + 1) * float(interval)
        parts.append(text)
        length += len(text) + 1
        prev_end = seg.end

    if parts:
        yield Paragraph(para_start, " ".join(parts), para_ts)


# =============================================================================
# Escrita dos formatos de saída (incremental, UTF-8)
# =============================================================================


def _safe_times(seg: Segment, min_duration: float = 0.2) -> tuple[float, float]:
    start = max(0.0, seg.start)
    end = seg.end if seg.end > start else start + min_duration
    return start, end


def write_txt(
    segments: Iterable[Segment],
    path: Path | str,
    header: TranscriptHeader,
    timestamp_mode: str = "segment",
) -> int:
    """Grava a transcrição em TXT. Retorna o número de parágrafos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    count = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        rule = "=" * 72
        fh.write("TRANSCRIÇÃO\n")
        fh.write(rule + "\n")
        for line in header.lines():
            fh.write(line + "\n")
        fh.write(rule + "\n\n")
        for para in build_paragraphs(segments, timestamp_mode):
            if timestamp_mode == "segment":
                fh.write(f"[{format_timestamp(para.start)}] {para.text}\n")
            else:
                if para.show_timestamp:
                    fh.write(f"[{format_timestamp(para.start)}]\n")
                fh.write(para.text + "\n\n")
            count += 1
        if count == 0:
            fh.write("(Nenhuma fala foi detectada no áudio.)\n")
        fh.flush()
    _replace(tmp, path)
    return count


def write_srt(segments: Iterable[Segment], path: Path | str) -> int:
    """Grava legendas SRT (UTF-8). Retorna o número de legendas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    index = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            index += 1
            start, end = _safe_times(seg)
            sub = srt.Subtitle(
                index=index,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                content=srt.make_legal_content(text),
            )
            fh.write(sub.to_srt())
    _replace(tmp, path)
    return index


def write_vtt(segments: Iterable[Segment], path: Path | str) -> int:
    """Grava legendas WebVTT (UTF-8). Retorna o número de cues."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    count = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("WEBVTT\n\n")
        for seg in segments:
            text = seg.text.strip().replace("-->", "->")
            text = re.sub(r"\n\s*\n", "\n", text)
            if not text:
                continue
            count += 1
            start, end = _safe_times(seg)
            fh.write(f"{count}\n{format_timestamp(start, 'vtt')} --> {format_timestamp(end, 'vtt')}\n{text}\n\n")
    _replace(tmp, path)
    return count


def _replace(tmp: Path, final: Path) -> None:
    for _ in range(40):
        try:
            os.replace(tmp, final)
            return
        except PermissionError:  # arquivo aberto em outro programa (ex.: Bloco de Notas)
            time.sleep(0.05)
    raise SubtitleError(
        f"Não foi possível salvar {final.name}: o arquivo pode estar aberto em outro programa."
    )


def export_all(
    segment_source: Callable[[], Iterable[Segment]],
    out_dir: Path | str,
    base_name: str,
    header: TranscriptHeader,
    timestamp_mode: str = "segment",
) -> dict[str, str]:
    """Gera TXT, SRT e VTT a partir de uma *fábrica* de segmentos.

    ``segment_source`` é chamada uma vez por formato (ex.:
    ``lambda: iter_segments_from_files(arquivos_dos_blocos)``), para que os
    segmentos sejam lidos do disco sob demanda, sem manter tudo na RAM.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "txt": out / f"{base_name}.txt",
        "srt": out / f"{base_name}.srt",
        "vtt": out / f"{base_name}.vtt",
    }
    write_txt(segment_source(), paths["txt"], header, timestamp_mode)
    write_srt(segment_source(), paths["srt"])
    write_vtt(segment_source(), paths["vtt"])
    return {k: str(v) for k, v in paths.items()}
