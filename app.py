"""Interface Streamlit do Transcritor de Vídeos Longos.

Esta interface NÃO transcreve nada: ela cria a tarefa (``config.json`` +
checkpoint inicial), inicia ``python -m src.worker`` com ``subprocess.Popen``
(via ``task_manager.start_worker``) e apenas LÊ o checkpoint JSON a cada ~1,5 s
com ``st.fragment(run_every=...)``. Fechar ou recarregar a página não
interrompe o worker; ao reabrir, a tarefa em andamento é reencontrada.

Execute com:  streamlit run app.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as components

from src import APP_NAME, LOGS_DIR, TEMP_DIR, TRANSCRIPTIONS_DIR, __version__, ensure_project_dirs
from src.audio_processor import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_SEARCH_WINDOW,
    DEFAULT_SILENCE_MIN_DURATION,
    DEFAULT_SILENCE_NOISE_DB,
    check_ffmpeg,
)
from src.downloader import SOURCE_LOCAL, SOURCE_TWITCH, SOURCE_YOUTUBE, validate_url
from src.file_picker import pick_file
from src.hardware import HardwareInfo, probe_hardware_subprocess, recommend_model, resolve_device
from src.subtitles import DEFAULT_SUBTITLE_MODE, SUBTITLE_MODES, language_display
from src.task_manager import (
    STEP_DONE,
    STEP_LABELS,
    STEP_ORDER,
    STEP_RUNNING,
    STEP_SKIPPED,
    active_task_ids,
    compute_overall,
    create_task,
    current_state,
    delete_task,
    elapsed_seconds,
    estimate_remaining_seconds,
    find_interrupted_tasks,
    list_tasks,
    read_log_tail,
    reconcile_task,
    request_cancel,
    resume_task,
    start_worker,
    worker_alive,
)
from src.utils import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_TIMESTAMP_MODE,
    DEVICE_CHOICES,
    LANGUAGES,
    MODELS,
    SUPPORTED_EXTENSIONS,
    TIMESTAMP_MODES,
    GPUError,
    InvalidInputError,
    TaskState,
    TranscriberError,
    classify_media_extension,
    ensure_disk_space,
    format_bytes,
    format_duration,
    format_timestamp,
    friendly_error,
    get_available_ram,
    open_folder,
    setup_logging,
    validate_local_file,
)

logger = logging.getLogger("app")

SOURCES: dict[str, str] = {
    "📁 Arquivo local": SOURCE_LOCAL,
    "▶️ YouTube": SOURCE_YOUTUBE,
    "🟣 Twitch (VOD)": SOURCE_TWITCH,
}

STATE_ICONS = {
    TaskState.PENDING: "🕒", TaskState.VALIDATING: "🔎", TaskState.DOWNLOADING: "⬇️",
    TaskState.EXTRACTING_AUDIO: "🎧", TaskState.SPLITTING: "✂️", TaskState.LOADING_MODEL: "🧠",
    TaskState.TRANSCRIBING: "📝", TaskState.FINALIZING: "📦", TaskState.COMPLETED: "✅",
    TaskState.CANCEL_REQUESTED: "⏳", TaskState.CANCELLED: "🚫", TaskState.FAILED: "❌",
}

PREVIEW_CHARS = 12_000          # prévia exibida na tela (o arquivo completo fica no disco)
COPY_MAX_BYTES = 2 * 1024 ** 2  # limite para o botão "copiar texto completo"

# =============================================================================
# Estilo
# =============================================================================

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1180px; }
.app-hero { padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1.2rem;
  background: linear-gradient(135deg, #2b1d6b 0%, #5b3cc4 55%, #8a5cff 100%); color: #fff; }
.app-hero h1 { margin: 0; font-size: 1.9rem; font-weight: 750; letter-spacing: -0.5px; }
.app-hero p { margin: .35rem 0 0 0; opacity: .88; font-size: 1rem; }
.badge { display:inline-block; padding: .15rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 600; }
.badge-ok { background:#12372a; color:#6ee7b7; } .badge-run { background:#1f2a52; color:#93b4ff; }
.badge-bad { background:#401b1b; color:#fca5a5; } .badge-warn { background:#3d3217; color:#fcd34d; }
.badge-idle { background:#2a2d3a; color:#b6bbcc; }
div[data-testid="stMetric"] { background: rgba(124,92,255,.08); border: 1px solid rgba(124,92,255,.22);
  padding: .7rem .9rem; border-radius: 14px; }
.step-line { padding: .18rem 0; font-size: .95rem; }
</style>
"""


def badge_for(state: TaskState) -> str:
    kind = {
        TaskState.COMPLETED: "ok", TaskState.FAILED: "bad", TaskState.CANCELLED: "warn",
        TaskState.CANCEL_REQUESTED: "warn", TaskState.PENDING: "idle",
    }.get(state, "run")
    return f'<span class="badge badge-{kind}">{STATE_ICONS.get(state, "")} {state.label}</span>'


# =============================================================================
# Recursos em cache
# =============================================================================


@st.cache_resource(show_spinner="Verificando o hardware do computador…")
def load_hardware() -> HardwareInfo:
    """Diagnóstico em processo isolado (protege a interface de falhas de DLL CUDA)."""
    return probe_hardware_subprocess()


@st.cache_data(ttl=20, show_spinner=False)
def ffmpeg_status() -> tuple[bool, str]:
    return check_ffmpeg()


def fresh_hardware() -> HardwareInfo:
    """Hardware em cache com a RAM disponível atualizada."""
    return replace(load_hardware(), ram_available_bytes=get_available_ram())


# =============================================================================
# Callbacks
# =============================================================================


def _cb_pick_file() -> None:
    """Abre o seletor nativo do Windows (processo isolado) e preenche o campo de caminho."""
    current = st.session_state.get("local_path", "")
    initial = str(Path(current).parent) if current and Path(current).parent.is_dir() else None
    try:
        chosen = pick_file(initial_dir=initial)
    except TranscriberError as exc:
        st.session_state["pick_error"] = exc.user_message
        return
    if chosen:
        st.session_state["local_path"] = chosen


def _cb_use_cpu() -> None:
    st.session_state["device_choice"] = "cpu"


def _cb_open_task(task_id: str) -> None:
    st.session_state["current_task"] = task_id


def _cb_back() -> None:
    st.session_state["current_task"] = None


def _cb_resume(task_id: str, device_override: Optional[str] = None) -> None:
    try:
        resume_task(task_id, device_override=device_override)
        st.session_state["current_task"] = task_id
        st.session_state.pop("task_error", None)
    except TranscriberError as exc:
        logger.error("Falha ao retomar %s: %s | %s", task_id, exc.user_message, exc.technical)
        st.session_state["task_error"] = exc.user_message
    except Exception as exc:  # noqa: BLE001
        logger.exception("Falha ao retomar %s", task_id)
        st.session_state["task_error"] = friendly_error(exc)


def _cb_ask_delete(task_id: str) -> None:
    st.session_state[f"confirm_delete_{task_id}"] = True


def _cb_delete(task_id: str) -> None:
    try:
        delete_task(task_id)
        if st.session_state.get("current_task") == task_id:
            st.session_state["current_task"] = None
    except TranscriberError as exc:
        st.session_state["task_error"] = exc.user_message
    except OSError as exc:
        st.session_state["task_error"] = friendly_error(exc)
    st.session_state.pop(f"confirm_delete_{task_id}", None)


def _cb_open_folder(path: str) -> None:
    try:
        open_folder(path)
    except Exception as exc:  # noqa: BLE001
        st.session_state["task_error"] = friendly_error(exc)


# =============================================================================
# Barra lateral
# =============================================================================


def render_sidebar(hw: HardwareInfo) -> None:
    with st.sidebar:
        st.markdown(f"### 🎙️ {APP_NAME}")
        st.caption(f"Versão {__version__} · processamento 100% local")

        ok, msg = ffmpeg_status()
        (st.success if ok else st.error)(("FFmpeg: " if ok else "") + msg)

        st.markdown("#### Seu computador")
        fresh = fresh_hardware()
        for line in fresh.summary_lines():
            st.caption(line)
        st.caption(f"Python {fresh.python_version}")
        if st.button("🔄 Verificar hardware novamente", use_container_width=True):
            load_hardware.clear()
            ffmpeg_status.clear()
            st.rerun()

        st.markdown("#### Pastas")
        st.button("📂 Abrir pasta das transcrições", use_container_width=True,
                  on_click=_cb_open_folder, args=(str(TRANSCRIPTIONS_DIR),))


# =============================================================================
# Nova transcrição
# =============================================================================


def render_new_task(hw: HardwareInfo) -> None:
    ss = st.session_state
    active = active_task_ids()
    if active:
        st.info("⏳ Já existe uma transcrição em andamento. Só uma tarefa roda por vez para não sobrecarregar o computador.")
        st.button("Ver andamento", type="primary", on_click=_cb_open_task, args=(active[0],))

    render_interrupted_banner()

    st.subheader("1 · Escolha a origem")
    source_label = st.radio("Origem do conteúdo", list(SOURCES), horizontal=True, key="source_choice",
                            label_visibility="collapsed")
    source_type = SOURCES[source_label]
    input_ok = False

    if source_type == SOURCE_LOCAL:
        c1, c2 = st.columns([5, 1.6], vertical_alignment="bottom")
        c1.text_input("Caminho completo do arquivo", key="local_path", placeholder=r"C:\Videos\gravacao.mp4",
                      help="Cole o caminho completo ou use o botão ao lado. O arquivo original nunca é alterado.")
        c2.button("📂 Selecionar arquivo…", on_click=_cb_pick_file, use_container_width=True)
        if ss.get("pick_error"):
            st.warning(ss.pop("pick_error"))
        text = ss.get("local_path", "")
        if text.strip():
            try:
                path = validate_local_file(text)
                kind = "vídeo" if classify_media_extension(path) == "video" else "áudio"
                st.success(f"✔ {path.name} — {kind} {path.suffix.upper().lstrip('.')} · {format_bytes(path.stat().st_size)}")
                input_ok = True
            except TranscriberError as exc:
                st.error(exc.user_message)
        st.caption("Formatos aceitos: " + ", ".join(sorted(e.lstrip(".").upper() for e in SUPPORTED_EXTENSIONS)))
    else:
        label = "Link do vídeo do YouTube" if source_type == SOURCE_YOUTUBE else "Link da gravação (VOD) da Twitch"
        example = "https://www.youtube.com/watch?v=..." if source_type == SOURCE_YOUTUBE else "https://www.twitch.tv/videos/123456789"
        st.text_input(label, key="url_input", placeholder=example)
        url_text = ss.get("url_input", "")
        if url_text.strip():
            try:
                v = validate_url(url_text)
                if v.source_type != source_type:
                    st.error("Este link é de outra plataforma. Troque a origem acima ou cole o link correto.")
                else:
                    st.success(f"✔ Link válido ({v.media_id})")
                    input_ok = True
            except TranscriberError as exc:
                st.error(exc.user_message)
        if source_type == SOURCE_YOUTUBE:
            st.radio("Legendas do YouTube", list(SUBTITLE_MODES), format_func=SUBTITLE_MODES.get,
                     key="subtitle_mode", index=list(SUBTITLE_MODES).index(DEFAULT_SUBTITLE_MODE),
                     help="Se houver legenda adequada, o texto é extraído sem baixar o áudio nem transcrever de novo.")
        with st.expander("Conteúdo que exige login (opcional)"):
            st.text_input("Caminho do arquivo cookies.txt", key="cookies_path", placeholder=r"C:\cookies\cookies.txt",
                          help="Somente de uma sessão sua e autorizada. O arquivo é copiado para a pasta da tarefa "
                               "e apagado ao final; o original não é modificado.")
            st.caption("Não é possível baixar conteúdo protegido por DRM, assinatura paga ou acesso restrito.")

    st.subheader("2 · Ajustes da transcrição")
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Idioma", list(LANGUAGES), format_func=LANGUAGES.get, key="lang_choice",
                 index=list(LANGUAGES).index(DEFAULT_LANGUAGE))
    c2.selectbox("Modelo", list(MODELS), format_func=lambda m: MODELS[m].label, key="model_choice",
                 index=list(MODELS).index(DEFAULT_MODEL))
    c3.selectbox("Dispositivo", list(DEVICE_CHOICES), format_func=DEVICE_CHOICES.get, key="device_choice")

    model = ss.get("model_choice", DEFAULT_MODEL)
    c2.caption(MODELS[model].description)
    c2.caption(f"💡 Sugestão para este computador: **{recommend_model(fresh_hardware())}**")

    device_blocked = False
    try:
        decision = resolve_device(ss.get("device_choice", "auto"), model, fresh_hardware(), allow_cpu_fallback=False)
        c3.caption(f"Será usado: **{decision.label}**")
        for w in decision.warnings:
            c3.warning(w)
    except GPUError as exc:
        device_blocked = True
        c3.error(exc.user_message)
        c3.button("Processar pela CPU", on_click=_cb_use_cpu, key="btn_use_cpu")

    with st.expander("ℹ️ Qual modelo escolher?"):
        st.markdown("Modelos maiores costumam ser mais precisos, mas exigem mais memória e tempo. "
                    "O modelo é baixado automaticamente no primeiro uso.")
        rows = ["| Modelo | Velocidade | RAM (CPU) | Quando usar |", "|---|---|---|---|"]
        for m in MODELS.values():
            rows.append(f"| `{m.name}` | {m.relative_speed} | ≈ {m.ram_gb_int8:.1f} GB | {m.description} |")
        st.markdown("\n".join(rows))

    st.selectbox("Marcações de tempo no TXT", list(TIMESTAMP_MODES), format_func=TIMESTAMP_MODES.get,
                 key="ts_mode", index=list(TIMESTAMP_MODES).index(DEFAULT_TIMESTAMP_MODE))

    with st.expander("⚙️ Opções avançadas"):
        a1, a2 = st.columns(2)
        a1.slider("Tamanho aproximado de cada bloco (minutos)", 5, 20, int(DEFAULT_CHUNK_SECONDS // 60), key="chunk_min")
        a1.slider("Janela de busca por pausas (± segundos)", 10, 120, int(DEFAULT_SEARCH_WINDOW), key="search_window")
        a1.checkbox("Cortar os blocos em pausas de silêncio (recomendado)", value=True, key="use_silence")
        a2.slider("Limite de silêncio (dB)", -60, -20, int(DEFAULT_SILENCE_NOISE_DB), key="noise_db",
                  help="Valores mais baixos tratam como silêncio apenas trechos mais silenciosos.")
        a2.slider("Duração mínima da pausa (segundos)", 0.2, 1.5, float(DEFAULT_SILENCE_MIN_DURATION), 0.1, key="min_pause")
        a2.slider("Beam size (precisão × velocidade)", 1, 10, 5, key="beam_size")
        st.checkbox("Manter arquivos temporários após concluir", key="keep_temp")
        st.checkbox("Se já existir um resultado com o mesmo nome, substituí-lo", key="overwrite_cb",
                    help="Por padrão uma nova pasta numerada é criada e nada é sobrescrito.")
        if ss.get("overwrite_cb"):
            st.checkbox("⚠️ Confirmo que o resultado anterior com esse nome pode ser substituído", key="overwrite_confirm")

    st.divider()
    ready = input_ok and not device_blocked and not active
    if st.button("🚀 Iniciar transcrição", type="primary", use_container_width=True, disabled=not ready):
        start_new_task(source_type, model)
    if not ready and not active:
        st.caption("Informe um arquivo ou link válido para habilitar o botão.")


def start_new_task(source_type: str, model: str) -> None:
    ss = st.session_state
    ok, msg = ffmpeg_status()
    if not ok:
        st.error(msg)
        return
    try:
        if source_type == SOURCE_LOCAL:
            path = validate_local_file(ss.get("local_path", ""))
            source, title = str(path), path.stem
        else:
            v = validate_url(ss.get("url_input", ""))
            if v.source_type != source_type:
                raise InvalidInputError("O link não corresponde à origem escolhida.")
            source, title = v.url, v.url
        ensure_disk_space(TEMP_DIR, 2 * 1024 ** 3, purpose="iniciar o processamento")

        device = ss.get("device_choice", "auto")
        resolve_device(device, model, fresh_hardware(), allow_cpu_fallback=False)  # GPUError se GPU indisponível

        overwrite = bool(ss.get("overwrite_cb") and ss.get("overwrite_confirm"))
        cookies = (ss.get("cookies_path") or "").strip() or None
        config = create_task(
            source_type,
            source,
            title,
            model=model,
            language=ss.get("lang_choice", DEFAULT_LANGUAGE),
            device=device,
            allow_cpu_fallback=device != "cuda",
            timestamp_mode=ss.get("ts_mode", DEFAULT_TIMESTAMP_MODE),
            subtitle_mode=ss.get("subtitle_mode", DEFAULT_SUBTITLE_MODE) if source_type == SOURCE_YOUTUBE else "ignore",
            cookies_file=cookies,
            chunk_seconds=float(ss.get("chunk_min", 10)) * 60,
            silence={
                "noise_db": float(ss.get("noise_db", DEFAULT_SILENCE_NOISE_DB)),
                "min_duration": float(ss.get("min_pause", DEFAULT_SILENCE_MIN_DURATION)),
                "search_window": float(ss.get("search_window", DEFAULT_SEARCH_WINDOW)),
            },
            use_silence_split=bool(ss.get("use_silence", True)),
            beam_size=int(ss.get("beam_size", 5)),
            overwrite_existing=overwrite,
            keep_temp=bool(ss.get("keep_temp", False)),
        )
        ss["current_task"] = config.task_id
        start_worker(config.task_id)
    except GPUError as exc:
        st.error(exc.user_message)
        return
    except TranscriberError as exc:
        logger.error("Não foi possível iniciar: %s | %s", exc.user_message, exc.technical)
        st.error(exc.user_message)
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro ao iniciar a tarefa")
        st.error(friendly_error(exc))
        return
    st.rerun()


# =============================================================================
# Tarefas anteriores / interrompidas
# =============================================================================


def render_interrupted_banner() -> None:
    try:
        interrupted = find_interrupted_tasks()[:3]
    except Exception:  # noqa: BLE001 - nunca derrubar a tela inicial por causa da lista
        logger.exception("Falha ao listar tarefas interrompidas")
        return
    if not interrupted:
        return
    st.warning(f"Há {len(interrupted)} tarefa(s) interrompida(s) que podem ser retomadas de onde pararam.")
    for t in interrupted:
        c1, c2, c3 = st.columns([5, 2, 2], vertical_alignment="center")
        c1.markdown(f"**{t.title[:80]}**  \n{t.completed_blocks} de {t.total_blocks or '?'} blocos concluídos")
        c2.button("▶ Retomar", key=f"banner_resume_{t.task_id}", on_click=_cb_resume, args=(t.task_id,))
        c3.button("Ver detalhes", key=f"banner_open_{t.task_id}", on_click=_cb_open_task, args=(t.task_id,))


def render_task_list() -> None:
    tasks = list_tasks()
    if err := st.session_state.pop("task_error", None):
        st.error(err)
    if not tasks:
        st.info("Nenhuma tarefa ainda. As transcrições que você iniciar aparecerão aqui.")
        return
    for t in tasks[:60]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([5, 3, 4], vertical_alignment="center")
            c1.markdown(f"**{t.title[:90]}**")
            c1.caption(f"{t.created_at[:16].replace('T', ' ')} · modelo {t.model} · {t.source_type}")
            c2.markdown(badge_for(t.state), unsafe_allow_html=True)
            c2.progress(min(1.0, t.progress), text=f"{t.progress * 100:.0f}% · {format_duration(t.elapsed)}")
            b1, b2, b3 = c3.columns(3)
            b1.button("Abrir", key=f"open_{t.task_id}", on_click=_cb_open_task, args=(t.task_id,))
            if t.resumable:
                b2.button("Retomar", key=f"res_{t.task_id}", on_click=_cb_resume, args=(t.task_id,))
            if not t.alive:
                if st.session_state.get(f"confirm_delete_{t.task_id}"):
                    c3.button("Confirmar exclusão", key=f"del_ok_{t.task_id}", type="primary",
                              on_click=_cb_delete, args=(t.task_id,),
                              help="Apaga só os arquivos temporários da tarefa. Resultados e arquivos originais não são tocados.")
                else:
                    b3.button("Excluir", key=f"del_{t.task_id}", on_click=_cb_ask_delete, args=(t.task_id,))


# =============================================================================
# Andamento
# =============================================================================


def _device_text(ck: dict[str, Any]) -> str:
    dev = ck.get("device") or {}
    return dev.get("label") or f"({DEVICE_CHOICES.get(ck.get('device_requested', 'auto'), 'automático')} — definido ao carregar o modelo)"


def _language_text(ck: dict[str, Any]) -> str:
    code = ck.get("language_detected") or ck.get("language_requested")
    if not code or code == "auto":
        return "detecção automática"
    text = language_display(code)
    prob = ck.get("language_probability")
    return f"{text} ({prob:.0%})" if prob else text


def render_progress(ck: dict[str, Any], task_id: str, state: TaskState) -> None:
    pct = compute_overall(ck)
    st.markdown(f"### {STATE_ICONS.get(state, '')} {state.label}")
    st.caption(ck.get("message") or "")
    st.progress(min(1.0, pct), text=f"{pct * 100:.1f}% concluído")

    total_blocks = int(ck.get("total_blocks") or 0)
    done_blocks = len(ck.get("completed_blocks") or [])
    remaining = estimate_remaining_seconds(ck)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Blocos concluídos", f"{done_blocks} / {total_blocks}" if total_blocks else "—")
    m2.metric("Tempo decorrido", format_duration(elapsed_seconds(ck)))
    m3.metric("Tempo restante (estimado)", format_duration(remaining) if remaining is not None else "calculando…")
    total_audio = ck.get("audio_duration")
    m4.metric("Áudio transcrito",
              f"{format_timestamp(ck.get('transcribed_seconds') or 0)} / {format_timestamp(total_audio)}" if total_audio else "—")

    d1, d2, d3, d4 = st.columns(4)
    d1.caption("Arquivo / vídeo")
    d1.write(ck.get("title", ""))
    d2.caption("Modelo")
    d2.write(ck.get("model", ""))
    d3.caption("Dispositivo")
    d3.write(_device_text(ck))
    d4.caption("Idioma")
    d4.write(_language_text(ck))

    st.markdown("#### Etapas")
    for step in STEP_ORDER:
        entry = (ck.get("steps") or {}).get(step, {})
        status, frac = entry.get("status"), float(entry.get("fraction") or 0.0)
        label = STEP_LABELS[step]
        if status == STEP_DONE:
            line = f"✅ {label}"
        elif status == STEP_RUNNING:
            line = f"⏳ **{label}** — {frac * 100:.0f}%"
        elif status == STEP_SKIPPED:
            line = f"⏭️ ~~{label}~~ (não necessária)"
        else:
            line = f"⬜ {label}"
        st.markdown(f'<div class="step-line">{line}</div>', unsafe_allow_html=False)

    for w in (ck.get("warnings") or [])[-3:]:
        st.warning(w)

    blocks = ck.get("blocks") or []
    if blocks:
        with st.expander(f"Blocos de áudio ({done_blocks}/{len(blocks)})"):
            status_names = {"done": "✅ concluído", "running": "⏳ em andamento", "pending": "⬜ aguardando"}
            st.dataframe(
                [{"Bloco": b["index"] + 1, "Início": format_timestamp(b["start"]), "Fim": format_timestamp(b["end"]),
                  "Situação": status_names.get(b["status"], b["status"])} for b in blocks],
                hide_index=True, use_container_width=True,
            )
    with st.expander("Mensagens recentes"):
        for msg in reversed((ck.get("messages") or [])[-12:]):
            icon = {"warning": "⚠️", "error": "❌"}.get(msg.get("level"), "•")
            st.caption(f"{icon} {msg['time'][11:19]} — {msg['text']}")

    cancel_pending = state == TaskState.CANCEL_REQUESTED
    if st.button("⛔ Cancelar processamento", key=f"cancel_{task_id}", disabled=cancel_pending,
                 help="Os blocos já transcritos são preservados e a tarefa pode ser retomada depois."):
        try:
            request_cancel(task_id)
            st.toast("Cancelamento solicitado. Encerrando em um ponto seguro…")
        except TranscriberError as exc:
            st.error(exc.user_message)
    if cancel_pending:
        st.info("Cancelamento solicitado. O processo será encerrado em um ponto seguro; os blocos concluídos ficam salvos.")


@st.fragment(run_every=1.5)
def progress_fragment(task_id: str) -> None:
    """Relê o checkpoint JSON a cada 1,5 s. Ao terminar, recarrega a página inteira para mostrar o resultado."""
    ck = reconcile_task(task_id)
    if ck is None:
        st.error("A tarefa não foi encontrada.")
        return
    if TaskState(ck.get("state", "pending")).is_terminal or not worker_alive(ck):
        st.rerun()
        return
    render_progress(ck, task_id, current_state(ck, task_id))


# =============================================================================
# Resultado / falha / interrupção
# =============================================================================


def _read_head(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_chars)
    except OSError:
        return ""


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def copy_button(text: str) -> None:
    """Botão que copia o texto para a área de transferência (com alternativa para navegadores restritivos)."""
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(
        f"""
<button id="b" style="width:100%;padding:.55rem 1rem;border-radius:10px;border:1px solid #7C5CFF;
 background:#1A1D29;color:#E6E8EF;font:600 14px sans-serif;cursor:pointer">📋 Copiar texto completo</button>
<script>
const text = {payload};
const btn = document.getElementById('b');
function fallback() {{
  const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta);
  ta.select(); let ok = false; try {{ ok = document.execCommand('copy'); }} catch (e) {{}}
  document.body.removeChild(ta); return ok;
}}
btn.onclick = async () => {{
  let ok = false;
  try {{ await navigator.clipboard.writeText(text); ok = true; }} catch (e) {{ ok = fallback(); }}
  btn.textContent = ok ? '✅ Copiado!' : '⚠️ Não foi possível copiar — use o botão de download';
  setTimeout(() => btn.textContent = '📋 Copiar texto completo', 2500);
}};
</script>""",
        height=58,
    )


def render_completed(ck: dict[str, Any], task_id: str) -> None:
    out = ck.get("output") or {}
    files: dict[str, str] = out.get("files") or {}
    st.success("✅ Transcrição concluída!")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tempo de processamento", format_duration(elapsed_seconds(ck)))
    m2.metric("Duração do conteúdo", format_timestamp(ck.get("audio_duration") or 0))
    m3.metric("Idioma", _language_text(ck))
    m4.metric("Pico de RAM do worker", f"{ck.get('peak_rss_mb', 0):.0f} MB")
    if ck.get("mode") == "subtitles":
        st.info("Texto obtido das legendas do YouTube — nenhum áudio foi baixado nem transcrito novamente.")
    st.caption(f"Pasta dos resultados: `{out.get('dir', '')}`")

    cols = st.columns(4)
    mimes = {"txt": "text/plain", "srt": "application/x-subrip", "vtt": "text/vtt"}
    for col, (ext, mime) in zip(cols, mimes.items()):
        path = files.get(ext)
        data = _read_bytes(path) if path else None
        if data is not None:
            col.download_button(f"⬇️ Baixar .{ext}", data=data, file_name=Path(path).name, mime=mime,
                                key=f"dl_{ext}_{task_id}", use_container_width=True)
        else:
            col.button(f".{ext} indisponível", disabled=True, key=f"dl_na_{ext}_{task_id}", use_container_width=True)
    if out.get("dir"):
        cols[3].button("📂 Abrir pasta", key=f"open_dir_{task_id}", use_container_width=True,
                       on_click=_cb_open_folder, args=(out["dir"],))

    txt = files.get("txt")
    if txt:
        head = _read_head(txt, PREVIEW_CHARS)
        with st.expander("👁️ Pré-visualizar o início da transcrição", expanded=True):
            st.text_area("Prévia", head, height=380, disabled=True, label_visibility="collapsed")
            try:
                size = Path(txt).stat().st_size
            except OSError:
                size = 0
            if size > len(head.encode("utf-8")):
                st.caption(f"Exibindo o começo do texto. O arquivo completo ({format_bytes(size)}) está na pasta dos resultados.")
        if st.checkbox("Mostrar o botão de copiar o texto completo", key=f"copy_cb_{task_id}"):
            try:
                size = Path(txt).stat().st_size
            except OSError:
                size = 0
            if 0 < size <= COPY_MAX_BYTES:
                copy_button(Path(txt).read_text(encoding="utf-8", errors="replace"))
            else:
                st.info("O texto é muito grande para copiar pelo navegador. Use o download ou abra a pasta.")

    st.button("➕ Nova transcrição", type="primary", on_click=_cb_back, key=f"new_{task_id}")


def render_problem(ck: dict[str, Any], task_id: str, state: TaskState) -> None:
    errors = ck.get("errors") or []
    last = errors[-1] if errors else None
    done, total = len(ck.get("completed_blocks") or []), int(ck.get("total_blocks") or 0)
    if err := st.session_state.pop("task_error", None):
        st.error(err)

    if state == TaskState.CANCELLED:
        st.warning("🚫 A tarefa foi cancelada.")
    elif state == TaskState.FAILED:
        st.error(f"❌ {last['message'] if last else 'A tarefa falhou.'}")
    else:
        st.info("A tarefa ainda não foi iniciada.")
    if total:
        st.progress(done / total, text=f"{done} de {total} blocos já transcritos e salvos")

    c1, c2 = st.columns(2)
    is_gpu_error = bool(last and last.get("code") == "gpu")
    if is_gpu_error:
        c1.button("▶ Retomar pela CPU", type="primary", on_click=_cb_resume, args=(task_id, "cpu"),
                  key=f"resume_cpu_{task_id}", use_container_width=True)
    label = "▶ Iniciar" if state == TaskState.PENDING else "▶ Retomar de onde parou"
    c2.button(label, type="primary" if not is_gpu_error else "secondary", on_click=_cb_resume, args=(task_id,),
              key=f"resume_{task_id}", use_container_width=True)
    st.button("➕ Nova transcrição", on_click=_cb_back, key=f"new_{task_id}")

    with st.expander("Detalhes técnicos"):
        for e in errors[-5:]:
            st.code(f"[{e.get('time', '')}] {e.get('type', '')}: {e.get('technical') or e.get('message', '')}", language="text")
        st.caption("Final do log do worker:")
        st.code(read_log_tail(task_id) or "(log vazio)", language="text")


def render_task_view(task_id: str) -> None:
    ck = reconcile_task(task_id)
    if ck is None:
        st.error("Esta tarefa não foi encontrada (pode ter sido excluída).")
        st.button("← Voltar", on_click=_cb_back)
        return
    st.button("← Voltar ao início", on_click=_cb_back, key="back_top",
              help="A transcrição continua em segundo plano; você pode voltar a ela pela aba 'Tarefas'.")
    state = TaskState(ck.get("state", "pending"))
    st.subheader(ck.get("title") or task_id)
    if state == TaskState.COMPLETED:
        render_completed(ck, task_id)
    elif state.is_terminal or not worker_alive(ck):
        render_problem(ck, task_id, state)
    else:
        progress_fragment(task_id)


# =============================================================================
# Página
# =============================================================================


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("current_task", None)
    ss.setdefault("local_path", "")
    ss.setdefault("device_choice", "auto")
    if not ss.get("_adopted"):
        ss["_adopted"] = True
        # Recarregar a página não interrompe o worker: reencontra a tarefa em andamento.
        active = active_task_ids()
        if active and ss["current_task"] is None:
            ss["current_task"] = active[0]


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🎙️", layout="wide", initial_sidebar_state="expanded")
    ensure_project_dirs()
    setup_logging(LOGS_DIR / "app.log", console=False)
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    hw = load_hardware()
    render_sidebar(hw)
    st.markdown(
        f'<div class="app-hero"><h1>🎙️ {APP_NAME}</h1>'
        "<p>Transcreva vídeos e gravações de horas, com fidelidade e privacidade: tudo é processado no seu computador.</p></div>",
        unsafe_allow_html=True,
    )

    task_id = st.session_state.get("current_task")
    if task_id:
        render_task_view(task_id)
        return
    tab_new, tab_tasks = st.tabs(["Nova transcrição", "Tarefas"])
    with tab_new:
        render_new_task(hw)
    with tab_tasks:
        render_task_list()


main()
