import os

os.environ["WATCHDOG_USE_POLLING"] = "true"

import pdf_sanitizer
import base64
import uuid
import json
import streamlit as st
import pandas as pd
import google.generativeai as genai
import toml
from datetime import datetime, timedelta
import pathlib

CACHE_FILE = ".visitas_cache.json"
EXPIRATION_HOURS = 12

# ---------- CONFIGURACIÓN ----------
st.set_page_config(
    page_title="Gestor de Visitas",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- API GEMINI ----------
def configure_api():
    api_key = st.secrets.get("API_KEY")
    if not api_key:
        local = pathlib.Path(__file__).parent / ".streamlit" / "secrets.toml"
        if local.exists():
            api_key = toml.load(local).get("API_KEY")
    if not api_key:
        st.error(
            "No se encontró API_KEY. Añádela en secrets.toml o en Streamlit Cloud."
        )
        return False
    genai.configure(api_key=api_key)
    return True


# ---------- UTILIDADES DE CACHÉ ----------
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
            if "ha_sido_editado" in p["df"].columns:
                p["df"]["ha_sido_editado"] = p["df"]["ha_sido_editado"].apply(
                    lambda x: "" if x is False or x is None else str(x)
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
@st.cache_resource(show_spinner="Procesando visita… Espera a que aparezca la tabla")
def extract_with_gemini(pdf_bytes: bytes) -> tuple[str, pd.DataFrame]:
    """
    Extrae datos de un PDF usando la API de Gemini.
    Asume que la API ya ha sido configurada.
    """
    # 🔐 Paso nuevo: borrar teléfonos
    pdf_bytes = pdf_sanitizer.sanitize_pdf(pdf_bytes, permanent=False)
    pdf_part = {"mime_type": "application/pdf", "data": pdf_bytes}
    prompt = """
Eres un experto OCR. El PDF contiene registros donde los nombres y apellidos aparecen en el siguiente formato:
APELLIDOS (pueden ser varios), Nombre
Devuelve **exactamente** este JSON:
{
  "title": "<TÍTULO EN UNA LÍNEA>",
  "rows": [
    {"Nombre": "...", "Apellido(s)": "...", "Participantes": "...", "Importe": "... €", "Reserva": "..."}
  ]
}
"""
    # Modelo rápido y multimodal
    model = genai.GenerativeModel("gemini-flash-lite-latest")
    response = model.generate_content([prompt, pdf_part])

    try:
        txt = response.text.strip().lstrip("```json").rstrip("```")
        data = json.loads(txt)
        df = pd.DataFrame(data["rows"])
        df.insert(0, "check", False)
        df["Notas"] = ""
        df["ha_sido_editado"] = ""  # Añadir columna de estado de edición
        return data["title"], df
    except Exception as e:
        st.error(f"Error procesando respuesta de Gemini: {e}")
        st.error("Respuesta cruda:\n" + response.text)
        return "Error en PDF", pd.DataFrame()


# ---------- CONFIGURAR API ----------
API_CONFIGURED = configure_api()

# ---------- INICIALIZACIÓN SESSION STATE Y CACHE ----------
if "pages" not in st.session_state or "page_counter" not in st.session_state:
    pages_cache, page_counter_cache = load_cache()
    pages_cache = clean_expired(pages_cache)
    if not pages_cache:
        page_counter_cache = 1
        pages_cache = [
            {
                "id": str(uuid.uuid4()),
                "name": "Visita 1",
                "df": None,
                "created_at": datetime.now().isoformat(),
            }
        ]
    st.session_state.pages = pages_cache
    st.session_state.page_counter = page_counter_cache
    st.session_state.current = 0

pages = st.session_state.pages
page_counter = st.session_state.page_counter
cur_idx = st.session_state.current

# Asegurar índice válido
if cur_idx >= len(pages):
    st.session_state.current = 0
    cur_idx = 0

# Crear página por defecto si la lista está vacía
if not pages:
    page_counter += 1
    new_page = {
        "id": str(uuid.uuid4()),
        "name": f"Visita {page_counter}",
        "df": None,
        "created_at": datetime.now().isoformat(),
    }
    pages.append(new_page)
    st.session_state.page_counter = page_counter

page = pages[cur_idx]


# ---------- SIDEBAR ----------
def extract_sidebar_label(title: str) -> str:
    parts = [p.strip() for p in title.split(",")]
    return ", ".join(parts[:3]) if len(parts) >= 3 else title


with st.sidebar:
    st.markdown("### Visitas")
    for i, p in enumerate(pages):
        label = extract_sidebar_label(p["name"])
        if p["df"] is not None:
            if (
                "ha_sido_editado" in p["df"].columns
                and (p["df"]["ha_sido_editado"] != "").any()
            ):
                label += " ✏️"
            if "check" in p["df"].columns and p["df"][~p["df"]["check"]].empty:
                label += " 🟢"
        col1, col2 = st.columns([0.85, 0.15], gap="small")
        with col1:
            if st.button(label, key=f"page_btn_{i}", use_container_width=True):
                st.session_state.current = i
                st.rerun()
        with col2:
            if st.button(
                "🗑️", key=f"del_btn_{i}", help="Borrar visita", use_container_width=True
            ):
                del pages[i]
                if not pages:
                    page_counter += 1
                    pages.append(
                        {
                            "id": str(uuid.uuid4()),
                            "name": f"Visita {page_counter}",
                            "df": None,
                            "created_at": datetime.now().isoformat(),
                        }
                    )
                    st.session_state.page_counter = page_counter
                if st.session_state.current >= len(pages):
                    st.session_state.current = max(0, len(pages) - 1)
                save_cache(pages, page_counter)
                st.rerun()
        st.markdown("---")

    if st.button("➕ Nueva Visita"):
        page_counter += 1
        pages.append(
            {
                "id": str(uuid.uuid4()),
                "name": f"Visita {page_counter}",
                "df": None,
                "created_at": datetime.now().isoformat(),
            }
        )
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
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        for key in ["pages", "page_counter", "current"]:
            st.session_state.pop(key, None)
        st.cache_resource.clear()
        st.rerun()

# ---------- ÁREA PRINCIPAL ----------
col1, col2 = st.columns([4, 1])
with col1:
    st.header(page["name"])
with col2:
    show_all = st.checkbox("Visibles todas las columnas", value=False)

if page["df"] is None:
    if API_CONFIGURED:
        pdf_file = st.file_uploader("Sube un Bookeo PDF", type=["pdf"])
        if pdf_file:
            title, df = extract_with_gemini(pdf_file.read())
            if not df.empty:
                page["name"] = title
                page["df"] = df
                save_cache(pages, page_counter)
                st.rerun()
    else:
        st.warning(
            "La funcionalidad de carga de PDF está deshabilitada hasta que se configure la API_KEY."
        )
else:
    df = page["df"].copy()  # Usar una copia para evitar mutaciones inesperadas
    if "check" not in df.columns:
        df.insert(0, "check", False)
    if "Notas" not in df.columns:
        df["Notas"] = ""
    if "ha_sido_editado" not in df.columns:
        df["ha_sido_editado"] = ""

    # Ensure ha_sido_editado is always a string and empty if no edits
    df["ha_sido_editado"] = df["ha_sido_editado"].apply(
        lambda x: "" if x is False or x is None else str(x)
    )

    # --- LÓGICA PARA RESALTAR FILAS ---
    # El lápiz aparece si la fila fue editada.
    df["✏️"] = df.apply(
        lambda row: f"✏️ {row['ha_sido_editado']}"
        if str(row["ha_sido_editado"]).strip()
        else "",
        axis=1,
    )

    # Inyectar CSS para aumentar la fuente, centrar el texto y permitir ancho automático.
    st.markdown(
        """
        <style>
        div[data-baseweb="table"] table td, div[data-baseweb="table"] table th {
            font-size: calc(1rem + 2pt);
            text-align: center;
            width: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    is_checked = df["check"].tolist()

    # ---------- CONFIGURACIÓN DE COLUMNAS ----------
    column_config = {
        "check": st.column_config.CheckboxColumn("Hecho"),
        "✏️": st.column_config.TextColumn(label="", disabled=True),
        "Notas": st.column_config.TextColumn(label="Notas", disabled=False),
        "ha_sido_editado": None,  # Ocultar esta columna
    }

    # Alternar columnas visibles según el estado del checkbox "Visibles todas las columnas"
    if show_all:
        desired_order = [
            "✏️",
            "check",
            "Nombre",
            "Apellido(s)",
            "Participantes",
            "Importe",
            "Reserva",
            "Notas",
        ]
    else:
        desired_order = [
            "✏️",
            "check",
            "Nombre",
            "Participantes",
            "Importe",
            "Reserva",
            "Notas",
        ]
    visible_columns = [col for col in desired_order if col in df.columns]

    for col in df.columns:
        if col in ["check", "✏️", "Notas", "ha_sido_editado"]:
            continue
        if col == "Participantes":
            column_config[col] = st.column_config.NumberColumn(
                disabled=False, format="%d", step=1, label=col
            )
        elif col in ["Apellido(s)"]:
            column_config[col] = st.column_config.TextColumn(disabled=False)
        else:
            column_config[col] = st.column_config.TextColumn(disabled=False)

    edited = st.data_editor(
        df,
        key=page["id"],
        column_config=column_config,
        column_order=visible_columns,
        hide_index=True,
        use_container_width=True,
    )

    # --- LÓGICA PARA DETECTAR CAMBIOS Y ACTUALIZAR ESTADO ---
    if not df.equals(edited):
        # Columnas a ignorar en la comparación (el emoji y el estado de edición)
        cols_to_ignore = ["✏️", "ha_sido_editado", "check"]
        df_comp = df.drop(columns=cols_to_ignore, errors="ignore")
        edited_comp = edited.drop(columns=cols_to_ignore, errors="ignore")

        # Forzar la conversión a string para una comparación más robusta
        # y encontrar las filas que han cambiado.
        diff_mask = df_comp.astype(str) != edited_comp.astype(str)

        # Iterar solo sobre las filas que han cambiado
        for idx in diff_mask[diff_mask.any(axis=1)].index:
            changed_cols = diff_mask.columns[diff_mask.loc[idx]].tolist()

            # Obtener las columnas ya editadas
            existing_edited_cols_str = df.loc[idx, "ha_sido_editado"]
            if isinstance(existing_edited_cols_str, str) and existing_edited_cols_str:
                existing_edited_cols = existing_edited_cols_str.split(", ")
            else:
                existing_edited_cols = []

            # Unir las nuevas con las existentes sin duplicados
            all_changed_cols = sorted(list(set(existing_edited_cols + changed_cols)))
            edited.loc[idx, "ha_sido_editado"] = ", ".join(all_changed_cols)

        # Guardar el DF modificado (sin la columna del emoji) y reiniciar la app
        page["df"] = edited.drop(columns=["✏️"])
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
