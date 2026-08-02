from __future__ import annotations

import re
from pathlib import Path

import fitz

from corpuscraft.pdf2code.models import DocumentExtraction, FontFace, StyleClass, StyleSheet, TextSpan

_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")
_FONT_FORMATS = {"ttf": "truetype", "otf": "opentype", "woff": "woff", "woff2": "woff2"}


def build_stylesheet(document: DocumentExtraction, font_dir: Path) -> StyleSheet:
    """Cluster text spans into named CSS classes, resolving each span's font
    to its actual embedded font file where possible (falling back to a
    name-based CSS stack for base-14/non-embedded fonts).
    """
    font_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(document.source_path)
    try:
        font_resources = _collect_font_resources(doc)
        font_faces: dict[str, FontFace] = {}
        classes: dict[tuple, StyleClass] = {}
        assignments: dict[str, str] = {}

        for page in document.pages:
            for index, primitive in enumerate(page.primitives):
                if not isinstance(primitive, TextSpan):
                    continue

                face = font_faces.get(primitive.font_family)
                if face is None:
                    face = _resolve_font(primitive.font_family, font_resources, doc, font_dir)
                    font_faces[primitive.font_family] = face

                key = (face.css_family, primitive.font_size, primitive.font_weight, primitive.italic, primitive.color)
                style_class = classes.get(key)
                if style_class is None:
                    style_class = StyleClass(
                        name=f"style{len(classes)}",
                        font_family=face.css_family,
                        font_size=primitive.font_size,
                        font_weight=primitive.font_weight,
                        italic=primitive.italic,
                        color=primitive.color,
                    )
                    classes[key] = style_class
                assignments[f"{page.page_index}:{index}"] = style_class.name

        embedded_faces = {f.css_family: f for f in font_faces.values() if f.font_path is not None}
        return StyleSheet(
            classes=list(classes.values()),
            font_faces=list(embedded_faces.values()),
            assignments=assignments,
        )
    finally:
        doc.close()


def render_stylesheet_css(stylesheet: StyleSheet) -> str:
    rules = []
    for face in stylesheet.font_faces:
        assert face.font_path is not None
        src = face.font_path.resolve().as_uri()
        format_hint = f" format('{face.font_format}')" if face.font_format else ""
        rules.append(
            f"@font-face {{ font-family: {_css_string(face.css_family)}; "
            f"src: url('{src}'){format_hint}; "
            f"font-weight: {face.weight}; font-style: {'italic' if face.italic else 'normal'}; }}"
        )
    for style_class in stylesheet.classes:
        rules.append(
            f".{style_class.name} {{ font-family: {_css_string(style_class.font_family)}, sans-serif; "
            f"font-size: {style_class.font_size:.2f}pt; font-weight: {style_class.font_weight}; "
            f"font-style: {'italic' if style_class.italic else 'normal'}; color: {style_class.color}; }}"
        )
    return "\n".join(rules)


def _css_string(name: str) -> str:
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _collect_font_resources(doc: fitz.Document) -> list[tuple[str, int]]:
    """(normalized_basefont, xref) for every font resource, deduped by xref."""
    seen: set[int] = set()
    resources = []
    for page in doc:
        for font_info in page.get_fonts(full=True):
            xref, basefont = font_info[0], font_info[3]
            if xref in seen:
                continue
            seen.add(xref)
            resources.append((_normalize(basefont), xref))
    return resources


def _normalize(name: str) -> str:
    name = _SUBSET_PREFIX.sub("", name)
    return re.sub(r"[\s\-,]", "", name).lower()


def _resolve_font(
    raw_family: str, font_resources: list[tuple[str, int]], doc: fitz.Document, font_dir: Path
) -> FontFace:
    weight = 700 if "bold" in raw_family.lower() else 400
    italic = "italic" in raw_family.lower() or "oblique" in raw_family.lower()

    target = _normalize(raw_family)
    # Exact matches always come first: a plain "Verdana" query must never
    # lose to a longer "VerdanaBold" substring match just because it's
    # longer -- that previously collapsed regular and bold onto the same
    # extracted file. Only fall back to substring matching (PyMuPDF's
    # basefont often carries a style suffix -- "DejaVu Sans Book" -- that
    # the span's font name doesn't have) when no exact match exists,
    # preferring the most specific (longest normalized) candidate there.
    exact = [xref for norm, xref in font_resources if norm == target]
    partial = sorted(
        (item for item in font_resources if item[0] != target and (target in item[0] or item[0] in target)),
        key=lambda item: -len(item[0]),
    )
    candidates = exact + [xref for _, xref in partial]

    for xref in candidates:
        _basename, ext, _ftype, buffer = doc.extract_font(xref)
        if buffer:
            font_path = font_dir / f"font{xref}.{ext}"
            font_path.write_bytes(buffer)
            return FontFace(
                css_family=raw_family,
                font_path=font_path,
                font_format=_FONT_FORMATS.get(ext),
                weight=weight,
                italic=italic,
            )

    return FontFace(css_family=raw_family, font_path=None, font_format=None, weight=weight, italic=italic)
