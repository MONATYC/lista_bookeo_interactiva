# pdf_sanitizer.py
"""
Anonimiza la columna cuyo encabezado contiene la palabra 'Teléfono'.
Compatible con PyMuPDF ≥1.18 (incluye fallback manual a búsqueda case-insensitive).
"""

import io
from pathlib import Path
from typing import List, Tuple, Union

import fitz  # PyMuPDF

Rect = fitz.Rect
BBox = Tuple[int, Rect]  # (page_index, rect)


def detect_keyword_boxes(
    pdf_source: Union[str, Path, bytes],
    keyword: str = "Teléfono",
    case_sensitive: bool = False,
) -> List[BBox]:
    """
    Devuelve [(page_index, rect), ...] con la posición exacta de `keyword`.
    Acepta ruta, Path o bytes.
    """
    boxes: List[BBox] = []
    compare_kw = keyword if case_sensitive else keyword.lower()

    # Abrimos desde ruta o desde flujo de bytes
    if isinstance(pdf_source, (str, Path)):
        doc = fitz.open(pdf_source)
    else:  # bytes
        doc = fitz.open(stream=pdf_source, filetype="pdf")

    for page_index, page in enumerate(doc):
        # Lista de palabras: [x0, y0, x1, y1, "palabra", block_no, line_no, word_no]
        for x0, y0, x1, y1, word, *_ in page.get_text("words"):
            compare_word = word if case_sensitive else word.lower()
            if compare_word == compare_kw:
                boxes.append((page_index, Rect(x0, y0, x1, y1)))

    doc.close()
    return boxes


def _redact_column_inplace(
    doc: fitz.Document,
    hits: List[BBox],
    padding: float = 2.0,
    permanent: bool = False,
) -> None:
    """
    Modifica `doc` en memoria tapando la columna donde aparezca cada hit.
    """
    for page_index, kw_rect in hits:
        page = doc[page_index]
        col_rect = Rect(
            kw_rect.x0 - padding,
            kw_rect.y0,  # Desde la misma línea del encabezado
            kw_rect.x1 + padding,
            page.rect.y1,  # Hasta el final de la página
        )

        if permanent:
            page.add_redact_annot(col_rect, fill=(0, 0, 0))
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        else:
            page.draw_rect(col_rect, fill=(0, 0, 0))


def sanitize_pdf(
    pdf_bytes: bytes,
    keyword: str = "Teléfono",
    case_sensitive: bool = False,
    padding: float = 2.0,
    permanent: bool = False,
) -> bytes:
    """
    Devuelve una copia anonimizada del PDF recibido en `pdf_bytes`.
    Sólo altera la columna bajo el encabezado `keyword`.
    """
    # 1) Localizamos el encabezado
    hits = detect_keyword_boxes(pdf_bytes, keyword, case_sensitive)
    if not hits:
        # Nada que tapar → devolvemos el original
        return pdf_bytes

    # 2) Abrimos el PDF y tapamos in-place
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    _redact_column_inplace(doc, hits, padding, permanent)

    # 3) Exportamos a bytes
    out = io.BytesIO()
    doc.save(out, garbage=4, deflate=True)
    doc.close()
    return out.getvalue()


# --- USO DE PRUEBA LOCAL (solo si se ejecuta directamente) ---
if __name__ == "__main__":
    src_pdf = "Bookeo_14septiembre2024_unido.pdf"
    dst_pdf = "Bookeo_14septiembre2024_unido_REDACT.pdf"

    with open(src_pdf, "rb") as f:
        pdf_bytes = f.read()

    redacted_bytes = sanitize_pdf(
        pdf_bytes, keyword="Teléfono", padding=4, permanent=False
    )

    with open(dst_pdf, "wb") as f:
        f.write(redacted_bytes)

    print(f"PDF protegido guardado en: {dst_pdf}")
