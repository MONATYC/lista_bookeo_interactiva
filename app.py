import uuid, os, base64, json
import streamlit as st
import pandas as pd
import google.generativeai as genai
import toml
from datetime import datetime, timedelta

CACHE_FILE = ".visitas_cache.json"
EXPIRATION_HOURS = 12

# ---------- CONFIGURACIÓN ----------
st.set_page_config(
    page_title="Gestor de Visitas", layout="wide", initial_sidebar_state="expanded"
)


# Cargar API_KEY desde secrets.toml
def get_api_key():
    try:
        secrets = toml.load("secrets.toml")
        return secrets["api"]["API_KEY"]
    except Exception:
        return None


def save_cache(pages, page_counter):
    serializable_pages = []
    for p in pages:
        page_copy = p.copy()
        if isinstance(page_copy.get("df"), pd.DataFrame):
            page_copy["df"] = page_copy["df"].to_dict("split")
        serializable_pages.append(page_copy)
    data = {"pages": serializable_pages, "page_counter": page_counter}
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return [], 0
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    pages = data.get("pages", [])
    for p in pages:
        df_data = p.get("df")
        if isinstance(df_data, dict):
            p["df"] = pd.DataFrame(
                df_data["data"], index=df_data["index"], columns=df_data["columns"]
            )
    return pages, data.get("page_counter", 0)


def clean_expired(pages):
    now = datetime.now()
    new_pages = []
    for p in pages:
        created = datetime.fromisoformat(p.get("created_at"))
        if now - created <= timedelta(hours=EXPIRATION_HOURS):
            new_pages.append(p)
    return new_pages


# ---------- FUNCIÓN DE EXTRACCIÓN ----------
@st.cache_resource(show_spinner="Procesando PDF con Gemini…")
def extract_with_gemini(pdf_bytes: bytes):
    api_key = get_api_key()
    if not api_key:
        st.error("No se encontró la API_KEY. Revisa tu archivo secrets.toml.")
        return "Error de configuración", pd.DataFrame()
    genai.configure(api_key=api_key)

    pdf_part = {"mime_type": "application/pdf", "data": pdf_bytes}
    prompt = """
Eres un experto OCR. Devuelve **exactamente** este JSON:
{
  "title": "<TÍTULO EN UNA LÍNEA>",
  "rows": [
    {"Cliente": "...", "Participantes": ..., "Importe": "... €", "Reserva": "..."}
  ]
}
"""
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = model.generate_content([prompt, pdf_part])

    try:
        txt = response.text.strip().lstrip("```json").rstrip("```")
        data = json.loads(txt)
        df = pd.DataFrame(data["rows"])
        df.insert(0, "check", False)
        return data["title"], df
    except Exception as e:
        st.error(f"Error procesando respuesta de Gemini: {e}")
        st.error(f"Respuesta cruda:\n{response.text}")
        return "Error en PDF", pd.DataFrame()


# ---------- INICIALIZACIÓN SESSION STATE Y CACHE ----------
if "pages" not in st.session_state or "page_counter" not in st.session_state:
    pages_cache, page_counter_cache = load_cache()
    pages_cache = clean_expired(pages_cache)
    if not pages_cache:
        page_counter_cache = 0
        # Crear página inicial vacía
        pages_cache = [
            {
                "id": str(uuid.uuid4()),
                "name": "Página 1",
                "df": None,
                "created_at": datetime.now().isoformat(),
            }
        ]
        page_counter_cache = 1
    st.session_state.pages = pages_cache
    st.session_state.page_counter = page_counter_cache
    st.session_state.current = 0

pages = st.session_state.pages
page_counter = st.session_state.page_counter
cur_idx = st.session_state.current

# Asegurar que current es válido tras limpieza
if cur_idx >= len(pages):
    st.session_state.current = 0
    cur_idx = 0

page = pages[cur_idx]


# ---------- SIDEBAR CON BORRADO ----------
def extract_sidebar_label(title: str) -> str:
    parts = [p.strip() for p in title.split(",")]
    if len(parts) >= 3:
        return ", ".join(parts[:3])
    return title


with st.sidebar:
    st.markdown("### Visitas")
    for i, p in enumerate(pages):
        label = extract_sidebar_label(p["name"])
        if p["df"] is not None and "check" in p["df"].columns:
            remaining = p["df"][~p["df"]["check"]]
            if remaining.empty:
                label += " 🟢"
        col1, col2 = st.columns([0.85, 0.15], gap="small")
        with col1:
            if st.button(
                label, key=f"page_btn_{i}", help=label, use_container_width=True
            ):
                st.session_state.current = i
                st.rerun()
        with col2:
            if st.button(
                "🗑️", key=f"del_btn_{i}", help="Borrar visita", use_container_width=True
            ):
                # Borra página y ajusta current si hace falta
                del pages[i]
                if st.session_state.current >= len(pages):
                    st.session_state.current = max(0, len(pages) - 1)
                save_cache(pages, page_counter)
                st.rerun()
        st.markdown("---")
    if st.button("➕ Nueva Visita"):
        page_counter += 1
        new_page = {
            "id": str(uuid.uuid4()),
            "name": f"Visita {page_counter}",
            "df": None,
            "created_at": datetime.now().isoformat(),
        }
        pages.append(new_page)
        st.session_state.page_counter = page_counter
        st.session_state.current = len(pages) - 1
        save_cache(pages, page_counter)
        st.rerun()

    st.markdown("---")
    st.error(
        "**Zona de Peligro**\n\n"
        "Al pulsar el siguiente botón se borrarán **todas** las visitas. "
        "Esta acción es irreversible."
    )
    if st.button("🗑️ Limpiar caché y reiniciar", use_container_width=True):
        # 1. Borrar el archivo de caché
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)

        # 2. Limpiar el estado de la sesión para forzar la reinicialización
        keys_to_clear = ["pages", "page_counter", "current"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # 3. Limpiar las cachés de funciones (como la de Gemini)
        st.cache_resource.clear()
        st.rerun()

# ---------- ÁREA PRINCIPAL ----------
st.header(page["name"])

if page["df"] is None:
    pdf_file = st.file_uploader("Sube un Bookeo PDF", type=["pdf"])
    if pdf_file:
        title, df = extract_with_gemini(pdf_file.read())
        page["name"] = title
        page["df"] = df
        save_cache(pages, page_counter)
        st.rerun()
else:
    df = page["df"]
    # Asegurarse de que exista la columna "check"
    if "check" not in df.columns:
        df.insert(0, "check", False)
    is_checked = df["check"].tolist()

    column_config = {"check": st.column_config.CheckboxColumn("Hecho")}
    for col in df.columns:
        if col == "check":
            continue
        if col == "Participantes":
            column_config[col] = st.column_config.NumberColumn(
                disabled=is_checked,
                format="%d",
                step=1,
                label=col,
            )
        else:
            column_config[col] = st.column_config.TextColumn(disabled=is_checked)

    edited = st.data_editor(
        df,
        key=page["id"],
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
    )

    if not df.equals(edited):
        page["df"] = edited
        save_cache(pages, page_counter)
        st.rerun()

    remaining = edited[~edited["check"]]
    if remaining.empty:
        st.success("✅ Visita completa")
    else:
        grupos = len(remaining)
        personas = int(remaining["Participantes"].astype(int).sum())
        st.warning(
            f"⏳ Faltan **{grupos}** grupos y un total de **{personas}** personas"
        )
