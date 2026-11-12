"""Worker de transcrição: processo INDEPENDENTE da interface.

Execução (feita por ``task_manager.start_worker``)::

    python -m src.worker --task-id AAAAMMDD-HHMMSS-xxxxxxxx

Regras:

* Não importa ``app.py`` nem o Streamlit e não usa ``st.session_state``.
* Recebe apenas o ID da tarefa e lê tudo de ``temp/tasks/<ID>/config.json``.
* Comunica-se por arquivos: grava ``checkpoint.json`` e ``manifest.json``;
  consulta ``cancel.json`` periodicamente (pontos seguros).
* Mantém ``worker.lock`` durante toda a execução (impede dois workers na mesma tarefa).
* stdout/stderr são redirecionados para ``logs/worker_<ID>.log`` pelo lançador.

Pipeline (idempotente — pode ser repetido após qualquer interrupção)::

    validar → [YouTube: consultar legendas → usar legenda] ou
              [baixar áudio] → converter (WAV 16 kHz mono) → detectar pausas e
              dividir em blocos → carregar modelo → transcrever bloco a bloco →
              organizar (TXT/SRT/VTT) → finalizar

Ordem de gravação ao concluir um bloco (a interrupção em qualquer ponto é segura):

1. ``segments/segments_NNNN.json`` (atômico);
2. ``manifest.json`` marca o bloco como ``done`` (fonte da verdade);
3. ``checkpoint.json`` atualizado (espelho para a interface).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psutil

from . import TRANSCRIPTIONS_DIR, ensure_project_dirs
from .audio_processor import (
    ChunkSpec,
    FFmpegInfo,
    SilenceSettings,
    build_chunk_plan,
    chunk_filename,
    extract_audio,
    find_ffmpeg,
    probe_media,
    split_wav,
    verify_chunk_continuity,
)
from .downloader import (
    SOURCE_LABELS,
    SOURCE_LOCAL,
    SOURCE_YOUTUBE,
    DownloadProgress,
    check_ytdlp,
    download_audio,
    download_subtitle,
    estimate_download_bytes,
    fetch_metadata,
    prepare_cookies_copy,
    validate_url,
)
from .hardware import DeviceDecision, detect_hardware, resolve_device
from .subtitles import (
    TranscriptHeader,
    choose_subtitle_track,
    export_all,
    iter_segments_from_files,
    parse_subtitle_file,
    save_segments_json,
)
from .task_manager import (
    STEP_DONE,
    STEP_PENDING,
    STEP_RUNNING,
    STEP_SKIPPED,
    CheckpointStore,
    add_error,
    add_message,
    compute_overall,
    elapsed_seconds,
    load_config,
    read_manifest,
    set_step,
    task_paths,
    write_manifest,
)
from .transcription import (
    TranscriptionEngine,
    TranscriptionSettings,
    normalize_language,
)
from .utils import (
    AudioProcessingError,
    CheckpointError,
    FileLock,
    InvalidInputError,
    SubtitleError,
    TaskCancelledError,
    TaskState,
    TranscriberError,
    current_process_identity,
    format_bytes,
    format_duration,
    format_timestamp,
    friendly_error,
    get_process_create_time,
    is_relative_to,
    is_valid_task_id,
    kill_process_tree,
    now_iso,
    sanitize_filename,
    setup_logging,
    unique_output_dir,
    validate_local_file,
)

logger = logging.getLogger("worker")

EXIT_OK, EXIT_FAILED, EXIT_BAD_ARGS, EXIT_ALREADY_RUNNING, EXIT_CANCELLED = 0, 1, 2, 3, 10

MODE_AUDIO, MODE_SUBTITLES = "audio", "subtitles"


class TaskRunner:
    """Executa uma tarefa do início ao fim (ou retoma de onde parou)."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.paths = task_paths(task_id)
        self.config = load_config(task_id)
        self.store = CheckpointStore(task_id, writer="worker")
        self._proc = psutil.Process()
        self._signal_received = False
        self._cancel_noted = False
        self._ffmpeg: Optional[FFmpegInfo] = None
        self._cookies: Optional[Path] = None
        self._local_path: Optional[Path] = None
        self._device: Optional[DeviceDecision] = None

    # ------------------------------------------------------------------
    # Cancelamento
    # ------------------------------------------------------------------

    def cancel_requested(self) -> bool:
        """``True`` se a interface pediu cancelamento (cancel.json) ou o sistema enviou um sinal."""
        requested = self._signal_received or self.paths.cancel.exists()
        if requested and not self._cancel_noted:
            self._cancel_noted = True
            try:
                self.store.mutate(self._apply_cancel_requested)
            except CheckpointError:
                logger.exception("Falha ao registrar cancel_requested")
        return requested

    @staticmethod
    def _apply_cancel_requested(ck: dict[str, Any]) -> None:
        if not TaskState(ck["state"]).is_terminal:
            ck["state"] = TaskState.CANCEL_REQUESTED.value
            add_message(ck, "Cancelamento solicitado. Encerrando em um ponto seguro…", "warning")

    def check_cancel(self) -> None:
        if self.cancel_requested():
            raise TaskCancelledError()

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            logger.warning("Sinal %s recebido; encerrando em um ponto seguro.", signum)
            self._signal_received = True

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handler)
                except (ValueError, OSError):
                    pass

    # ------------------------------------------------------------------
    # Escrita no checkpoint (apenas o worker)
    # ------------------------------------------------------------------

    def _finalize_ck(self, ck: dict[str, Any]) -> None:
        ck["overall_progress"] = compute_overall(ck)
        try:
            rss = self._proc.memory_info().rss / (1024 * 1024)
            ck["peak_rss_mb"] = round(max(float(ck.get("peak_rss_mb") or 0.0), rss), 1)
        except psutil.Error:
            pass

    def say(self, text: str, level: str = "info") -> None:
        logger.log(logging.WARNING if level == "warning" else logging.INFO, text)

        def _apply(ck: dict[str, Any]) -> None:
            add_message(ck, text, level)
            if level == "warning":
                ck.setdefault("warnings", []).append(text)
                del ck["warnings"][:-30]

        self.store.mutate(_apply)

    def begin(self, step: str, state: TaskState, message: str) -> None:
        """Inicia uma etapa: estado + mensagem; etapas anteriores ainda 'running' viram 'done'."""
        logger.info("[%s] %s", step, message)

        def _apply(ck: dict[str, Any]) -> None:
            for name, entry in ck["steps"].items():
                if name != step and entry["status"] == STEP_RUNNING:
                    set_step(ck, name, STEP_DONE)
            if ck["steps"][step]["status"] != STEP_DONE:
                set_step(ck, step, STEP_RUNNING, 0.0)
            if not TaskState(ck["state"]).is_terminal and ck["state"] != TaskState.CANCEL_REQUESTED.value:
                ck["state"] = state.value
            ck["step"] = step
            add_message(ck, message)
            self._finalize_ck(ck)

        self.store.mutate(_apply)

    def finish_step(self, step: str) -> None:
        def _apply(ck: dict[str, Any]) -> None:
            set_step(ck, step, STEP_DONE)
            self._finalize_ck(ck)

        self.store.mutate(_apply)

    def skip_steps(self, *steps: str) -> None:
        def _apply(ck: dict[str, Any]) -> None:
            for s in steps:
                set_step(ck, s, STEP_SKIPPED, 0.0)
            self._finalize_ck(ck)

        self.store.mutate(_apply)

    def progress(
        self,
        step: str,
        fraction: Optional[float],
        message: Optional[str] = None,
        *,
        extra: Optional[Any] = None,
        force: bool = False,
    ) -> None:
        """Atualiza a fração REAL de uma etapa (gravações frequentes são limitadas a ~1 por 0,7 s)."""

        def _apply(ck: dict[str, Any]) -> None:
            if fraction is not None:
                set_step(ck, step, fraction=fraction)
            if message:
                ck["message"] = message
            if extra:
                extra(ck)
            self._finalize_ck(ck)

        if force:
            self.store.mutate(_apply)
        else:
            self.store.mutate_throttled(_apply)

    # -- processos filhos (FFmpeg / yt-dlp) -------------------------------

    def _child_started(self, pid: int) -> None:
        created = get_process_create_time(pid)

        def _apply(ck: dict[str, Any]) -> None:
            ck.setdefault("child_pids", []).append({"pid": pid, "create_time": created, "started": now_iso()})

        self.store.mutate(_apply)

    def _child_exited(self, pid: int) -> None:
        def _apply(ck: dict[str, Any]) -> None:
            ck["child_pids"] = [c for c in ck.get("child_pids", []) if c.get("pid") != pid]

        self.store.mutate(_apply)

    def _on_retry(self, attempt: int, delay: float, exc: BaseException) -> None:
        self.say(f"Falha temporária de conexão (tentativa {attempt}). Nova tentativa em {delay:.0f} s…", "warning")

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def run(self) -> int:
        lock = FileLock(self.paths.worker_lock, timeout=0.0, stale_after=10 ** 9)
        try:
            lock.acquire()
        except CheckpointError:
            logger.error("Já existe um worker ativo para a tarefa %s.", self.task_id)
            return EXIT_ALREADY_RUNNING

        code = EXIT_FAILED
        try:
            self._install_signal_handlers()
            self._register_start()
            self._pipeline()
            code = EXIT_OK
        except TaskCancelledError:
            self._mark_cancelled()
            code = EXIT_CANCELLED
        except BaseException as exc:  # noqa: BLE001 - qualquer falha deve ficar registrada
            if isinstance(exc, TranscriberError):
                logger.error("Tarefa falhou: %s | %s", exc.user_message, exc.technical)
            else:
                logger.exception("Erro inesperado na tarefa %s", self.task_id)
            self._mark_failed(exc)
            code = EXIT_FAILED
        finally:
            self._cleanup_cookies()
            # Nenhum FFmpeg/yt-dlp pode sobrar depois do worker.
            try:
                kill_process_tree(os.getpid(), include_parent=False)
            except Exception:  # noqa: BLE001
                logger.exception("Falha ao encerrar processos filhos")
            lock.release()
        return code

    def _register_start(self) -> None:
        ident = current_process_identity()
        manifest = read_manifest(self.task_id)

        def _apply(ck: dict[str, Any]) -> None:
            worker = dict(ck.get("worker") or {})
            worker.update(pid=ident["pid"], create_time=ident["create_time"], started_at=now_iso())
            ck["worker"] = worker
            ck["run_started_epoch"] = time.time()
            ck["started_at"] = ck.get("started_at") or now_iso()
            ck["finished_at"] = None
            ck["state"] = TaskState.VALIDATING.value
            ck["child_pids"] = []
            for entry in ck["steps"].values():
                if entry["status"] == STEP_RUNNING:  # etapa interrompida: será refeita
                    entry["status"], entry["fraction"] = STEP_PENDING, 0.0
            self._sync_from_manifest(ck, manifest)
            resumed = int(ck.get("resume_count") or 0) > 0
            add_message(ck, "Retomando o processamento…" if resumed else "Processamento iniciado.")
            self._finalize_ck(ck)

        self.store.mutate(_apply)

    @staticmethod
    def _sync_from_manifest(ck: dict[str, Any], manifest: dict[str, Any]) -> None:
        """Espelha no checkpoint o que o manifesto (fonte da verdade) diz sobre os blocos."""
        chunks = manifest.get("chunks") or []
        if not chunks:
            return
        ck["mode"] = manifest.get("mode") or ck.get("mode")
        ck["total_blocks"] = len(chunks)
        ck["completed_blocks"] = [c["index"] for c in chunks if c.get("status") == "done"]
        ck["blocks"] = [
            {"index": c["index"], "start": c["start"], "end": c["end"], "status": c.get("status", "pending")}
            for c in chunks
        ]
        ck["transcribed_seconds"] = round(sum(c["duration"] for c in chunks if c.get("status") == "done"), 3)
        ck["audio_duration"] = round(sum(c["duration"] for c in chunks), 3)

    def _stop_clock(self, ck: dict[str, Any]) -> None:
        if ck.get("run_started_epoch"):
            ck["elapsed_before"] = elapsed_seconds(ck)
            ck["run_started_epoch"] = None

    def _mark_cancelled(self) -> None:
        reason = "sinal do sistema" if self._signal_received and not self.paths.cancel.exists() else "usuário"

        def _apply(ck: dict[str, Any]) -> None:
            self._stop_clock(ck)
            ck["state"] = TaskState.CANCELLED.value
            ck["finished_at"] = now_iso()
            ck["current_block"] = None
            ck["child_pids"] = []
            done, total = len(ck.get("completed_blocks") or []), int(ck.get("total_blocks") or 0)
            extra = f" {done} de {total} blocos foram preservados." if total else ""
            add_message(ck, f"Tarefa cancelada ({reason}).{extra} Você pode retomá-la depois.")
            self._finalize_ck(ck)

        self._safe_mutate(_apply)

    def _mark_failed(self, exc: BaseException) -> None:
        message = exc.user_message if isinstance(exc, TranscriberError) else friendly_error(exc)

        def _apply(ck: dict[str, Any]) -> None:
            self._stop_clock(ck)
            add_error(ck, exc)
            ck["state"] = TaskState.FAILED.value
            ck["finished_at"] = now_iso()
            ck["current_block"] = None
            ck["child_pids"] = []
            add_message(ck, message, "error")
            self._finalize_ck(ck)

        self._safe_mutate(_apply)

    def _safe_mutate(self, fn: Any) -> None:
        try:
            self.store.mutate(fn)
        except Exception:  # noqa: BLE001 - nunca mascarar o erro original
            logger.exception("Não foi possível gravar o estado final no checkpoint")

    def _cleanup_cookies(self) -> None:
        try:
            shutil.rmtree(self.paths.work_dir, ignore_errors=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _chunk_path(self, index: int) -> Path:
        return self.paths.chunks_dir / chunk_filename(index)

    def _pipeline(self) -> None:
        cfg = self.config
        manifest = read_manifest(self.task_id)

        self.begin("validate", TaskState.VALIDATING, "Verificando o arquivo ou link…")
        self._ffmpeg = find_ffmpeg()
        self.check_cancel()

        if cfg.source_type == SOURCE_LOCAL:
            self._validate_local(manifest)
        else:
            self._resolve_remote(manifest)

        mode = manifest.get("mode", MODE_AUDIO)
        if mode == MODE_SUBTITLES:
            self.skip_steps("download", "convert", "split", "load_model", "transcribe")
            self._finalize_output(manifest, MODE_SUBTITLES)
            return

        chunks = self._prepare_audio(manifest)
        self.check_cancel()
        self._transcribe(manifest, chunks)
        self._finalize_output(manifest, MODE_AUDIO)

    # -- 1. validação ----------------------------------------------------

    def _validate_local(self, manifest: dict[str, Any]) -> None:
        cfg = self.config
        self._local_path = validate_local_file(cfg.source)
        info = probe_media(self._local_path, self._ffmpeg)
        manifest.update(mode=MODE_AUDIO, title=cfg.title, duration=info.duration)
        write_manifest(self.task_id, manifest)

        def _apply(ck: dict[str, Any]) -> None:
            ck["audio_duration"] = info.duration
            ck["mode"] = MODE_AUDIO
            ck["title"] = cfg.title
            set_step(ck, "validate", STEP_DONE)
            for s in ("subtitles", "download"):
                set_step(ck, s, STEP_SKIPPED, 0.0)
            add_message(
                ck,
                f"Arquivo válido: {self._local_path.name} — {format_bytes(info.size_bytes)}"
                + (f", duração {format_duration(info.duration)}." if info.duration else "."),
            )
            self._finalize_ck(ck)

        self.store.mutate(_apply)

    def _resolve_remote(self, manifest: dict[str, Any]) -> None:
        cfg = self.config
        validated = validate_url(cfg.source)
        if validated.source_type != cfg.source_type:
            raise InvalidInputError("O link não corresponde ao tipo de origem escolhido.")
        check_ytdlp()
        self._cookies = prepare_cookies_copy(cfg.cookies_file, self.paths.work_dir)
        manifest["url"] = validated.url

        if manifest.get("mode") in (MODE_AUDIO, MODE_SUBTITLES) and manifest.get("title"):
            self.say("Informações do vídeo já obtidas anteriormente; continuando.")
            self.finish_step("validate")
            if cfg.source_type == SOURCE_YOUTUBE:
                self.finish_step("subtitles")
            return

        meta = fetch_metadata(
            validated.url,
            source_type=validated.source_type,
            cookies=self._cookies,
            cancel_check=self.cancel_requested,
            on_retry=self._on_retry,
        )
        title = meta.get("title") or cfg.title
        duration = meta.get("duration")
        manifest.update(
            mode=MODE_AUDIO, title=title, duration=duration, expected_bytes=estimate_download_bytes(meta)
        )
        write_manifest(self.task_id, manifest)

        def _apply(ck: dict[str, Any]) -> None:
            ck["title"] = title
            ck["audio_duration"] = duration
            set_step(ck, "validate", STEP_DONE)
            add_message(ck, f"Vídeo encontrado: {title}" + (f" ({format_duration(duration)})." if duration else "."))
            self._finalize_ck(ck)

        self.store.mutate(_apply)
        self.check_cancel()

        if cfg.source_type != SOURCE_YOUTUBE:
            return  # Twitch: sem etapa de legendas

        self.begin("subtitles", TaskState.VALIDATING, "Consultando legendas disponíveis…")
        selection = choose_subtitle_track(meta, cfg.subtitle_mode)
        self.say(selection.message, "info" if selection.track or selection.portuguese_available else "warning")
        if selection.track is not None:
            try:
                vtt = download_subtitle(
                    validated.url,
                    selection.track,
                    self.paths.work_dir / "subtitles",
                    source_type=validated.source_type,
                    cookies=self._cookies,
                    cancel_check=self.cancel_requested,
                    on_retry=self._on_retry,
                    ffmpeg_dir=self._ffmpeg.bin_dir if self._ffmpeg else None,
                )
                segments = parse_subtitle_file(vtt)
                if not segments:
                    raise SubtitleError("A legenda está vazia.")
            except TaskCancelledError:
                raise
            except TranscriberError as exc:
                self.say(f"Não foi possível usar a legenda ({exc.user_message}). O áudio será transcrito.", "warning")
            else:
                save_segments_json(
                    self.paths.subtitles_segments_file,
                    segments,
                    meta={"source": "subtitles", "track": selection.track.to_dict()},
                )
                end = max(s.end for s in segments)
                manifest.update(
                    mode=MODE_SUBTITLES,
                    subtitle=selection.track.to_dict(),
                    duration=duration or end,
                    n_segments=len(segments),
                )
                write_manifest(self.task_id, manifest)

                def _apply_sub(ck: dict[str, Any]) -> None:
                    ck["mode"] = MODE_SUBTITLES
                    ck["language_detected"] = selection.track.base_lang
                    ck["audio_duration"] = duration or end
                    ck["transcribed_seconds"] = duration or end
                    add_message(ck, f"Legenda extraída ({len(segments)} trechos). Nenhum áudio será baixado.")
                    self._finalize_ck(ck)

                self.store.mutate(_apply_sub)
        self.finish_step("subtitles")

    # -- 3–5. áudio, conversão e divisão em blocos -----------------------

    def _download_progress(self, p: DownloadProgress) -> None:
        text = f"Baixando o áudio… {format_bytes(p.downloaded_bytes)}"
        if p.total_bytes:
            text += f" de {format_bytes(p.total_bytes)}"
        self.progress("download", p.fraction, text)

    def _prepare_audio(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        """Garante que existam os arquivos de todos os blocos pendentes. Retorna os blocos do manifesto."""
        cfg = self.config
        planned = manifest.get("chunks") or []
        missing = [c for c in planned if c.get("status") != "done" and not self._chunk_path(c["index"]).is_file()]
        if planned and manifest.get("split_done") and not missing:
            self.say("Blocos de áudio já preparados; pulando download, conversão e divisão.")
            if cfg.source_type == SOURCE_LOCAL:
                self.finish_step("convert")
            else:
                for step in ("download", "convert"):
                    self.finish_step(step)
            self.finish_step("split")
            return planned

        # --- origem do áudio ---
        if cfg.source_type == SOURCE_LOCAL:
            assert self._local_path is not None
            source_audio = self._local_path
        else:
            self.begin("download", TaskState.DOWNLOADING, "Baixando somente o áudio…")
            source_audio = download_audio(
                manifest.get("url", cfg.source),
                self.paths.audio_dir,
                source_type=cfg.source_type,
                cookies=self._cookies,
                expected_bytes=manifest.get("expected_bytes"),
                progress_cb=self._download_progress,
                cancel_check=self.cancel_requested,
                on_process_start=self._child_started,
                on_process_exit=self._child_exited,
                on_retry=self._on_retry,
                ffmpeg_dir=self._ffmpeg.bin_dir if self._ffmpeg else None,
            )
            manifest["source_audio"] = str(source_audio)
            write_manifest(self.task_id, manifest)
            self.finish_step("download")
        self.check_cancel()

        # --- conversão para WAV 16 kHz mono ---
        self.begin("convert", TaskState.EXTRACTING_AUDIO, "Convertendo o áudio para 16 kHz mono…")

        def _convert_progress(fraction: Optional[float], seconds: float) -> None:
            msg = f"Convertendo o áudio… {format_duration(seconds)} processados"
            self.progress("convert", fraction, msg)

        wav = extract_audio(
            source_audio,
            self.paths.audio_dir / "work.wav",
            ffmpeg=self._ffmpeg,
            duration=manifest.get("duration"),
            progress_cb=_convert_progress,
            cancel_check=self.cancel_requested,
            on_process_start=self._child_started,
            on_process_exit=self._child_exited,
        )
        manifest["wav"] = wav.to_dict()
        manifest["duration"] = wav.duration
        write_manifest(self.task_id, manifest)
        self.finish_step("convert")
        self.check_cancel()

        # --- pausas e divisão em blocos ---
        self.begin("split", TaskState.SPLITTING, "Detectando pausas e dividindo o áudio em blocos…")
        if planned:
            chunks = [ChunkSpec.from_dict(c) for c in planned]
            verify_chunk_continuity(chunks, wav.frames)
            self.say(f"Plano de blocos reutilizado ({len(chunks)} blocos).")
        else:
            chunks = build_chunk_plan(
                wav.path,
                chunk_seconds=cfg.chunk_seconds,
                settings=SilenceSettings.from_dict(cfg.silence),
                ffmpeg=self._ffmpeg,
                use_silence=cfg.use_silence_split,
                cancel_check=self.cancel_requested,
                progress_cb=lambda n, total: self.progress(
                    "split", 0.5 * n / max(total, 1), f"Detectando pausas… bloco {n} de ~{total}"
                ),
            )
            manifest["plan"] = {
                "chunk_seconds": cfg.chunk_seconds,
                "silence": SilenceSettings.from_dict(cfg.silence).to_dict(),
                "use_silence_split": cfg.use_silence_split,
                "created_at": now_iso(),
            }
            manifest["chunks"] = [{**c.to_dict(), "status": "pending"} for c in chunks]
            write_manifest(self.task_id, manifest)
            reasons = [c.cut_reason for c in chunks]
            self.say(
                f"Áudio dividido em {len(chunks)} blocos "
                f"({reasons.count('silence')} cortes em pausas, {reasons.count('nominal')} no limite nominal)."
            )

            def _sync(ck: dict[str, Any]) -> None:
                self._sync_from_manifest(ck, manifest)
                self._finalize_ck(ck)

            self.store.mutate(_sync)

        statuses = {c["index"]: c.get("status") for c in manifest["chunks"]}
        todo = [c for c in chunks if statuses.get(c.index) != "done"]
        split_wav(
            wav.path,
            todo,
            self.paths.chunks_dir,
            cancel_check=self.cancel_requested,
            progress_cb=lambda i, n: self.progress(
                "split", 0.5 + 0.5 * i / max(n, 1), f"Gravando blocos de áudio… {i} de {n}"
            ),
        )
        manifest["split_done"] = True
        write_manifest(self.task_id, manifest)
        self.finish_step("split")
        self._free_audio_temp(manifest)
        return manifest["chunks"]

    def _free_audio_temp(self, manifest: dict[str, Any]) -> None:
        """Apaga WAV de trabalho e áudio baixado (somente dentro da pasta da tarefa)."""
        if self.config.keep_temp:
            return
        candidates = [(manifest.get("wav") or {}).get("path"), manifest.get("source_audio")]
        for item in candidates:
            if not item:
                continue
            p = Path(item)
            # Nunca apagar o arquivo original do usuário: só o que está dentro da pasta da tarefa.
            if is_relative_to(p, self.paths.root):
                try:
                    p.unlink()
                except OSError:
                    pass

    # -- 6–7. modelo e transcrição ---------------------------------------

    def _on_device_change(self, decision: DeviceDecision, reason: str) -> None:
        self._device = decision

        def _apply(ck: dict[str, Any]) -> None:
            ck["device"] = decision.to_dict() | {"label": decision.label}
            ck.setdefault("warnings", []).append(f"A GPU falhou e o processamento continuou pela CPU: {reason[:160]}")
            self._finalize_ck(ck)

        self.store.mutate(_apply)

    def _transcribe(self, manifest: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        cfg = self.config
        pending = [c for c in chunks if c.get("status") != "done"]
        total_seconds = sum(c["duration"] for c in chunks)
        done_seconds = sum(c["duration"] for c in chunks if c.get("status") == "done")

        if not pending:
            self.say("Todos os blocos já estavam transcritos.")
            self.skip_steps("load_model")
            self.finish_step("transcribe")
            return

        # --- dispositivo e modelo ---
        self.begin("load_model", TaskState.LOADING_MODEL, "Verificando o hardware e carregando o modelo…")
        hardware = detect_hardware()
        decision = resolve_device(cfg.device, cfg.model, hardware, allow_cpu_fallback=cfg.allow_cpu_fallback)
        self._device = decision

        def _apply_device(ck: dict[str, Any]) -> None:
            ck["device"] = decision.to_dict() | {"label": decision.label}
            ck["warnings"] = list(dict.fromkeys((ck.get("warnings") or []) + decision.warnings))[-30:]
            self._finalize_ck(ck)

        self.store.mutate(_apply_device)
        for w in decision.warnings:
            logger.warning(w)

        language = normalize_language(cfg.language)
        locked = (manifest.get("language") or {}).get("code")
        if language is None and locked:
            language = locked

        settings = TranscriptionSettings(model=cfg.model, language=language, beam_size=cfg.beam_size)
        engine = TranscriptionEngine(
            settings,
            decision,
            allow_cpu_fallback=cfg.allow_cpu_fallback,
            on_message=lambda t: self.say(t),
            on_device_change=self._on_device_change,
        )
        try:
            engine.load(self.cancel_requested)
            self.finish_step("load_model")

            # --- blocos ---
            self.begin("transcribe", TaskState.TRANSCRIBING, f"Transcrevendo {len(pending)} bloco(s)…")

            def _start_run(ck: dict[str, Any]) -> None:
                ck["transcribe_run"] = {"started_epoch": time.time(), "audio_seconds": 0.0}
                ck["transcribed_seconds"] = round(done_seconds, 3)
                set_step(ck, "transcribe", fraction=done_seconds / total_seconds if total_seconds else 0.0)
                if cfg.language != "auto":
                    ck["language_detected"] = cfg.language
                elif language:
                    ck["language_detected"] = language
                self._finalize_ck(ck)

            self.store.mutate(_start_run)

            run_seconds = 0.0
            for chunk in chunks:
                if chunk.get("status") == "done":
                    continue
                self.check_cancel()
                idx = chunk["index"]
                path = self._chunk_path(idx)
                if not path.is_file():
                    raise AudioProcessingError(
                        "Um arquivo temporário de áudio desapareceu. Retome a tarefa para recriá-lo.",
                        technical=f"bloco {idx}: {path}",
                    )
                label = (
                    f"Transcrevendo bloco {idx + 1} de {len(chunks)} "
                    f"({format_timestamp(chunk['start'])}–{format_timestamp(chunk['end'])})…"
                )

                def _begin_block(ck: dict[str, Any], idx: int = idx, label: str = label) -> None:
                    ck["current_block"] = idx
                    for b in ck.get("blocks", []):
                        if b["index"] == idx:
                            b["status"] = "running"
                    add_message(ck, label)
                    self._finalize_ck(ck)

                self.store.mutate(_begin_block)

                def _block_progress(sec: float, dur: float, base: float = done_seconds, run_base: float = run_seconds) -> None:
                    def _extra(ck: dict[str, Any]) -> None:
                        ck["transcribed_seconds"] = round(base + sec, 3)
                        ck["transcribe_run"]["audio_seconds"] = round(run_base + sec, 3)

                    self.progress(
                        "transcribe",
                        (base + sec) / total_seconds if total_seconds else 0.0,
                        extra=_extra,
                    )

                result = engine.transcribe_file(
                    path,
                    offset=chunk["start"],
                    chunk_duration=chunk["duration"],
                    language=language,
                    progress_cb=_block_progress,
                    cancel_check=self.cancel_requested,
                )

                # 1) segmentos do bloco (atômico)
                save_segments_json(
                    self.paths.segments_file(idx),
                    result.segments,
                    meta={
                        "index": idx,
                        "offset": chunk["start"],
                        "duration": chunk["duration"],
                        "language": result.language,
                        "language_probability": result.language_probability,
                        "elapsed": round(result.elapsed, 2),
                    },
                )
                # 2) manifesto (fonte da verdade)
                chunk.update(
                    status="done",
                    segments_file=str(self.paths.segments_file(idx)),
                    n_segments=len(result.segments),
                    completed_at=now_iso(),
                    elapsed=round(result.elapsed, 2),
                    language=result.language,
                )
                if (
                    normalize_language(cfg.language) is None
                    and language is None
                    and result.speech_detected
                    and (result.language_probability or 0.0) >= 0.5
                ):
                    language = result.language  # trava o idioma detectado para os demais blocos
                    manifest["language"] = {"code": result.language, "probability": result.language_probability}
                write_manifest(self.task_id, manifest)

                done_seconds += chunk["duration"]
                run_seconds += chunk["duration"]

                # 3) checkpoint (espelho para a interface)
                def _end_block(ck: dict[str, Any], idx: int = idx, done: float = done_seconds, run: float = run_seconds) -> None:
                    self._sync_from_manifest(ck, manifest)
                    ck["current_block"] = None
                    ck["transcribed_seconds"] = round(done, 3)
                    ck["transcribe_run"]["audio_seconds"] = round(run, 3)
                    set_step(ck, "transcribe", fraction=done / total_seconds if total_seconds else 1.0)
                    if language and normalize_language(cfg.language) is None:
                        ck["language_detected"] = language
                        ck["language_probability"] = (manifest.get("language") or {}).get("probability")
                    add_message(
                        ck,
                        f"Bloco {idx + 1} de {len(chunks)} concluído "
                        f"({len(result.segments)} trechos, {result.elapsed:.0f} s de processamento).",
                    )
                    self._finalize_ck(ck)

                self.store.mutate(_end_block)
                if not cfg.keep_temp:
                    try:
                        path.unlink()
                    except OSError:
                        pass
        finally:
            engine.close()
        self.finish_step("transcribe")

    # -- 8–9. organização e finalização ----------------------------------

    def _finalize_output(self, manifest: dict[str, Any], mode: str) -> None:
        cfg = self.config
        self.check_cancel()
        self.begin("organize", TaskState.FINALIZING, "Organizando o texto e gerando TXT, SRT e VTT…")

        if mode == MODE_SUBTITLES:
            files = [self.paths.subtitles_segments_file]
        else:
            files = [self.paths.segments_file(c["index"]) for c in manifest.get("chunks", [])]
        missing = [f.name for f in files if not f.is_file()]
        if not files or missing:
            raise AudioProcessingError(
                "Faltam trechos transcritos para montar o resultado. Retome a tarefa.",
                technical=f"arquivos ausentes: {missing}",
            )

        ck = self.store.read()
        title = ck.get("title") or cfg.title
        out_dir_str = (ck.get("output") or {}).get("dir")
        if out_dir_str:
            out_dir = Path(out_dir_str)
        elif cfg.overwrite_existing:
            out_dir = TRANSCRIPTIONS_DIR / sanitize_filename(title)
        else:
            out_dir = unique_output_dir(TRANSCRIPTIONS_DIR, title)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.store.mutate(lambda c: c.__setitem__("output", {"dir": str(out_dir), "files": {}}))

        language = (manifest.get("language") or {}).get("code") or ck.get("language_detected")
        probability = (manifest.get("language") or {}).get("probability")
        if cfg.language != "auto" and mode == MODE_AUDIO:
            language, probability = cfg.language, None
        device_label = (ck.get("device") or {}).get("label")
        if mode == MODE_SUBTITLES:
            track = manifest.get("subtitle") or {}
            kind = "adicionada pelo criador" if track.get("kind") == "manual" else "gerada automaticamente"
            method = f"Legendas do YouTube ({kind}, idioma {track.get('lang', '?')}) — sem nova transcrição"
        else:
            method = "Transcrição automática do áudio (faster-whisper)"

        header = TranscriptHeader(
            title=title,
            source_type=SOURCE_LABELS.get(cfg.source_type, cfg.source_type),
            source=cfg.source,
            method=method,
            processed_at=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            language=language,
            language_probability=probability,
            model=cfg.model if mode == MODE_AUDIO else None,
            device=device_label if mode == MODE_AUDIO else None,
            duration=manifest.get("duration"),
        )
        # Os segmentos são lidos do disco, um bloco por vez, a cada formato.
        outputs = export_all(
            lambda: iter_segments_from_files(files),
            out_dir,
            sanitize_filename(title),
            header,
            cfg.timestamp_mode,
        )
        self.finish_step("organize")

        self.begin("finish", TaskState.FINALIZING, "Finalizando…")

        def _done(c: dict[str, Any]) -> None:
            self._stop_clock(c)
            set_step(c, "finish", STEP_DONE)
            c["state"] = TaskState.COMPLETED.value
            c["finished_at"] = now_iso()
            c["output"] = {"dir": str(out_dir), "files": outputs}
            c["current_block"] = None
            c["child_pids"] = []
            if language:
                c["language_detected"] = language
            c["overall_progress"] = 1.0
            add_message(c, f"Concluído! Arquivos salvos em {out_dir}")
            self._finalize_ck(c)
            c["overall_progress"] = 1.0

        self.store.mutate(_done)

        if not cfg.keep_temp:
            for d in (self.paths.audio_dir, self.paths.chunks_dir):
                shutil.rmtree(d, ignore_errors=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Worker de transcrição (processo independente).")
    parser.add_argument("--task-id", required=True, help="Identificador da tarefa em temp/tasks/")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    setup_logging(None, console=True)
    ensure_project_dirs()

    if not is_valid_task_id(args.task_id):
        logger.error("ID de tarefa inválido: %r", args.task_id)
        return EXIT_BAD_ARGS
    try:
        runner = TaskRunner(args.task_id)
    except Exception:  # noqa: BLE001
        logger.exception("Não foi possível carregar a tarefa %s", args.task_id)
        return EXIT_BAD_ARGS
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
