"""Tablas y figuras del informe, listas para llevarse a un paper.

El informe es un HTML autocontenido: las figuras son PNG a 120 dpi dentro de
base64 y las tablas son texto HTML. Para un manuscrito sirve poco, porque una
figura necesita ser vectorial (o de 300 dpi) y una tabla necesita ser datos, no
una captura. Este modulo toma lo que el informe ya generó y lo deja en tres
formas sin tocar cada punto donde se dibuja algo:

* **Numeracion.** Cada tabla y cada figura recibe su número ("Tabla 3",
  "Figura 5") y su leyenda, en el idioma del informe.
* **Enlaces dentro del HTML.** Bajo cada tabla, CSV y LaTeX; bajo cada figura,
  SVG. Sirven cuando el HTML viaja solo.
* **Carpeta de exportacion** junto al informe, con todo en todos los formatos.

Como se engancha:

    _fig_to_b64(fig)            -> capturar(fig, b64)    (cada figura de matplotlib)
    generate_report(...)        -> iniciar() ... detener() alrededor del armado
    html final                  -> procesar(html, ruta)  (numera, enlaza, exporta)

Las tablas no se registran una por una: se leen del HTML final, que es donde
ya estan con sus titulos y sus numeros de seccion resueltos. Asi una tabla
nueva en el informe sale exportable sin tocar este archivo.

No hay dependencias nuevas: el XLSX se escribe a mano (un zip de XML) en vez de
pedir openpyxl, que no esta en requirements.txt.
"""
from __future__ import annotations

import base64
import bisect
import csv
import hashlib
import html as _html
import io
import re
import shutil
import unicodedata
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle

from .i18n import tr, idioma

# (nombre del panel, columnas, filas)
Panel = Tuple[str, List[str], List[list]]

SUBCARPETA_TABLAS = "tablas"
SUBCARPETA_FIGURAS = "figuras"
SUBCARPETA_DATOS = "datos"


@dataclass
class _Figura:
    titulo: str
    svg: bytes = b""
    pdf: bytes = b""
    png: bytes = b""
    paneles: List[Panel] = field(default_factory=list)


# Registro activo mientras se arma un informe. None = fuera de un informe, y
# capturar() no hace nada: las figuras se siguen dibujando como siempre.
_registro: Optional[Dict[str, _Figura]] = None
_pesado: bool = False


def iniciar(pesado: bool = True) -> None:
    """Empieza a registrar las figuras que se dibujen.

    ``pesado`` decide si además se renderizan SVG, PDF y PNG a 300 dpi (lo que
    cuesta). Sin él solo se guarda el título, suficiente para numerar.
    """
    global _registro, _pesado
    _registro = {}
    _pesado = pesado


def detener() -> None:
    global _registro, _pesado
    _registro = None
    _pesado = False


def _clave(b64: str) -> str:
    return hashlib.sha1(b64.encode("ascii")).hexdigest()


def adjuntar_datos(fig, columnas: Sequence[str], filas: Sequence[Sequence],
                   nombre: str = "") -> None:
    """Fija los datos de una figura cuando extraerlos del dibujo se queda corto.

    Un histograma dibujado solo contiene las alturas de las barras; quien lo
    quiere rehacer en Origin o en Python necesita los valores crudos. Un boxplot
    es peor: sus datos dibujados son los bigotes, no las mediciones.
    """
    paneles = getattr(fig, "_polyx_paneles", [])
    paneles.append((nombre, list(columnas), [list(f) for f in filas]))
    fig._polyx_paneles = paneles


def _titulo_de(fig) -> str:
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text():
        return sup.get_text()
    for ax in fig.axes:
        if ax.get_title():
            return ax.get_title()
    return ""


def capturar(fig, b64: str) -> None:
    """Registra una figura de matplotlib. Llamar antes de cerrarla."""
    if _registro is None:
        return
    f = _Figura(titulo=_titulo_de(fig))
    if _pesado:
        # svg.fonttype "none" deja el texto como texto: se edita en Inkscape o
        # Illustrator. pdf.fonttype 42 incrusta TrueType en vez de Type 3, que
        # las revistas rechazan con frecuencia.
        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            for fmt, dpi in (("svg", None), ("pdf", None), ("png", 300)):
                buf = io.BytesIO()
                kw = {"dpi": dpi} if dpi else {}
                fig.savefig(buf, format=fmt, bbox_inches="tight", **kw)
                setattr(f, fmt, buf.getvalue())
        # Los textos de los ejes solo existen tras dibujar; por eso va despues.
        f.paneles = getattr(fig, "_polyx_paneles", None) or _paneles_de(fig)
    _registro[_clave(b64)] = f


# ────────────────────────────────────────────────────────────────────
# Datos de una figura, leidos del propio dibujo
# ────────────────────────────────────────────────────────────────────
def _etiqueta_cercana(ax, centro: float, semiancho: float) -> str:
    """Texto del tick del eje X que cae bajo una barra, si lo hay."""
    for pos, lab in zip(ax.get_xticks(), ax.get_xticklabels()):
        txt = lab.get_text()
        if txt and abs(pos - centro) <= max(semiancho, 1e-9) * 1.01:
            return txt.replace("\n", " ")
    return ""


def _paneles_de(fig) -> List[Panel]:
    paneles: List[Panel] = []
    for i, ax in enumerate(fig.axes, start=1):
        nombre = ax.get_title() or f"panel{i}"

        barras = [p for p in ax.patches
                  if isinstance(p, Rectangle) and p.get_width() and p.get_height()]
        if barras:
            filas = []
            for p in barras:
                x0, w = p.get_x(), p.get_width()
                filas.append([_etiqueta_cercana(ax, x0 + w / 2, abs(w) / 2),
                              x0, x0 + w, p.get_height()])
            paneles.append((nombre, ["categoria", "x_inicio", "x_fin", "valor"], filas))

        series = []
        for k, ln in enumerate(ax.lines, start=1):
            xs, ys = ln.get_data()
            if len(xs) == 0:
                continue
            rot = ln.get_label()
            rot = f"serie{k}" if (not rot or rot.startswith("_")) else rot
            series += [[rot, float(x), float(y)] for x, y in zip(xs, ys)]
        if series:
            paneles.append((nombre, ["serie", "x", "y"], series))

        puntos = []
        for k, col in enumerate(ax.collections, start=1):
            if not isinstance(col, PathCollection):
                continue
            off = np.asarray(col.get_offsets())
            if off.ndim != 2 or off.shape[1] != 2:
                continue
            rot = col.get_label()
            rot = f"serie{k}" if (not rot or rot.startswith("_")) else rot
            puntos += [[rot, float(x), float(y)] for x, y in off]
        if puntos:
            paneles.append((nombre, ["serie", "x", "y"], puntos))

        for im in ax.images:
            m = np.asarray(im.get_array())
            if m.ndim != 2:
                continue
            cols = [t.get_text() for t in ax.get_xticklabels()][:m.shape[1]]
            fils = [t.get_text() for t in ax.get_yticklabels()][:m.shape[0]]
            cols += [str(j) for j in range(len(cols), m.shape[1])]
            fils += [str(j) for j in range(len(fils), m.shape[0])]
            filas = [[fils[r]] + [float(v) for v in m[r]] for r in range(m.shape[0])]
            paneles.append((nombre, [ax.get_ylabel() or "fila"] + cols, filas))
    return paneles


# ────────────────────────────────────────────────────────────────────
# Tablas, leidas del HTML
# ────────────────────────────────────────────────────────────────────
class _LectorTabla(HTMLParser):
    """Filas y celdas de una tabla: texto plano, sin etiquetas anidadas."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.filas: List[List[str]] = []
        self._celda: Optional[List[str]] = None
        self._fila: Optional[List[str]] = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._fila = []
        elif tag in ("td", "th") and self._fila is not None:
            self._celda = []
            span = dict(attrs).get("colspan")
            self._span = int(span) if span and span.isdigit() else 1
        elif tag == "br" and self._celda is not None:
            self._celda.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._celda is not None and self._fila is not None:
            txt = re.sub(r"\s+", " ", "".join(self._celda)).strip()
            self._fila.append(txt)
            # Una celda combinada ocupa varias columnas: se rellena para que
            # todas las filas tengan el mismo ancho en el CSV.
            self._fila.extend([""] * (getattr(self, "_span", 1) - 1))
            self._celda = None
        elif tag == "tr" and self._fila is not None:
            if self._fila:
                self.filas.append(self._fila)
            self._fila = None

    def handle_data(self, data):
        if self._celda is not None:
            self._celda.append(data)


def _filas_de_tabla(html_tabla: str) -> List[List[str]]:
    lector = _LectorTabla()
    lector.feed(html_tabla)
    filas = lector.filas
    ancho = max((len(f) for f in filas), default=0)
    return [f + [""] * (ancho - len(f)) for f in filas]


_RE_NUM = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?$")


def _es_numero(s: str) -> bool:
    return bool(_RE_NUM.match(s.strip()))


def _csv_bytes(columnas: Sequence[str], filas: Sequence[Sequence]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(columnas)
    w.writerows(filas)
    # BOM: sin el, Excel en Windows abre un CSV UTF-8 con los acentos rotos.
    return buf.getvalue().encode("utf-8-sig")


_TEX_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex(s: str) -> str:
    return "".join(_TEX_ESCAPES.get(c, c) for c in s)


def _latex_tabla(filas: List[List[str]], leyenda: str, etiqueta: str) -> bytes:
    """Tabla de tres lineas (booktabs), como la piden las revistas."""
    if not filas:
        return b""
    cab, cuerpo = filas[0], filas[1:]
    # Una columna es numerica si todo su cuerpo lo es (o es un guion): se alinea
    # a la derecha. La primera va siempre a la izquierda, es la etiqueta.
    alin = ["l"]
    for j in range(1, len(cab)):
        vals = [f[j] for f in cuerpo if f[j] not in ("", "—", "-")]
        alin.append("r" if vals and all(_es_numero(v) for v in vals) else "l")
    lineas = [
        "% Requiere \\usepackage{booktabs}",
        "\\begin{table}[htbp]", "\\centering",
        f"\\caption{{{_tex(leyenda)}}}", f"\\label{{{etiqueta}}}",
        f"\\begin{{tabular}}{{{''.join(alin)}}}", "\\toprule",
        " & ".join(_tex(c) for c in cab) + " \\\\", "\\midrule",
    ]
    lineas += [" & ".join(_tex(c) for c in f) + " \\\\" for f in cuerpo]
    lineas += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lineas).encode("utf-8")


# ── XLSX minimo, sin openpyxl ───────────────────────────────────────
def _col_xlsx(n: int) -> str:
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _xml(s: str) -> str:
    return _html.escape(s, quote=False)


def _xlsx_bytes(hojas: List[Tuple[str, List[List[str]]]]) -> bytes:
    """Libro con una hoja por tabla. Los numeros van como numeros, no como texto."""
    nombres: List[str] = []
    for nom, _ in hojas:
        limpio = re.sub(r"[\[\]\*\?/\\:]", " ", nom)[:31].strip() or "Hoja"
        base, k = limpio, 2
        while limpio in nombres:
            limpio = f"{base[:28]} {k}"
            k += 1
        nombres.append(limpio)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                             for i in range(1, len(hojas) + 1))
                   + "</Types>")
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                   "</Relationships>")
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                   + "".join(f'<sheet name="{_xml(n)}" sheetId="{i}" r:id="rId{i}"/>'
                             for i, n in enumerate(nombres, start=1))
                   + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                             for i in range(1, len(hojas) + 1))
                   + "</Relationships>")
        for i, (_, filas) in enumerate(hojas, start=1):
            cuerpo = []
            for r, fila in enumerate(filas, start=1):
                celdas = []
                for c, v in enumerate(fila):
                    ref = f"{_col_xlsx(c)}{r}"
                    if v == "":
                        continue
                    if r > 1 and _es_numero(v):
                        celdas.append(f'<c r="{ref}"><v>{v}</v></c>')
                    else:
                        celdas.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{_xml(v)}</t></is></c>')
                cuerpo.append(f'<row r="{r}">{"".join(celdas)}</row>')
            z.writestr(f"xl/worksheets/sheet{i}.xml",
                       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                       '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                       f'<sheetData>{"".join(cuerpo)}</sheetData></worksheet>')
    return buf.getvalue()


# ────────────────────────────────────────────────────────────────────
# Nombres
# ────────────────────────────────────────────────────────────────────
def _slug(texto: str, maximo: int = 40) -> str:
    s = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:maximo].strip("_")


def _sin_numeracion(titulo_html: str) -> str:
    """Texto de un encabezado sin etiquetas ni el '3.2' del inicio."""
    t = re.sub(r"<[^>]+>", "", titulo_html)
    t = _html.unescape(re.sub(r"\s+", " ", t)).strip()
    return re.sub(r"^\d+(\.\d+)*\.?\s*", "", t)


def _nombre(prefijo: str, n: int, titulo: str) -> str:
    base = f"{prefijo}{n:02d}"
    s = _slug(titulo)
    return f"{base}_{s}_{idioma().upper()}" if s else f"{base}_{idioma().upper()}"


# ────────────────────────────────────────────────────────────────────
# Armado: numerar, enlazar y escribir la carpeta
# ────────────────────────────────────────────────────────────────────
_RE_TABLA = re.compile(r"<table class='data'[^>]*>.*?</table>", re.S)
_RE_IMG = re.compile(r"<img src='data:image/png;base64,([A-Za-z0-9+/=]+)'[^>]*>")
_RE_CAPTION = re.compile(r"\s*<div class='caption'>(.*?)</div>", re.S)
_RE_ENCAB = re.compile(r"<h([234])\b[^>]*>(.*?)</h\1>", re.S)


@dataclass
class _ItemTabla:
    n: int
    titulo: str
    filas: List[List[str]]
    nombre: str


@dataclass
class _ItemFigura:
    n: int
    leyenda: str
    fig: _Figura
    nombre: str


def _enlace(nombre: str, mime: str, datos: bytes, texto: str) -> str:
    b64 = base64.b64encode(datos).decode("ascii")
    return f"<a download='{nombre}' href='data:{mime};base64,{b64}'>{texto}</a>"


def procesar(html: str, ruta_informe: Path, exportar: bool = True) -> str:
    """Numera tablas y figuras del HTML, añade enlaces y escribe la carpeta.

    Devuelve el HTML ya modificado. Con ``exportar`` falso solo numera: el
    informe sale igual de ordenado pero sin enlaces ni carpeta.
    """
    etq_tabla, etq_fig = tr("Tabla"), tr("Figura")
    encabezados = [(m.start(), _sin_numeracion(m.group(2))) for m in _RE_ENCAB.finditer(html)]
    pos_enc = [p for p, _ in encabezados]
    registro = _registro or {}

    # Cada edicion es (inicio, fin, texto). Una insercion tiene inicio == fin.
    ediciones: List[Tuple[int, int, str]] = []
    tablas: List[_ItemTabla] = []
    figuras: List[_ItemFigura] = []

    # ── Tablas ──
    for m in _RE_TABLA.finditer(html):
        filas = _filas_de_tabla(m.group(0))
        if len(filas) < 2:
            continue
        k = bisect.bisect_right(pos_enc, m.start()) - 1
        titulo = encabezados[k][1] if k >= 0 else ""
        n = len(tablas) + 1
        item = _ItemTabla(n, titulo, filas, _nombre(etq_tabla, n, titulo))
        tablas.append(item)
        rotulo = f"{etq_tabla} {n}." + (f" {_html.escape(titulo)}" if titulo else "")
        ediciones.append((m.start(), m.start(), f"<div class='tabcap'>{rotulo}</div>"))
        if exportar:
            leyenda = f"{etq_tabla} {n}. {titulo}".strip()
            enlaces = " · ".join([
                _enlace(item.nombre + ".csv", "text/csv;charset=utf-8",
                        _csv_bytes(filas[0], filas[1:]), "CSV"),
                _enlace(item.nombre + ".tex", "text/plain;charset=utf-8",
                        _latex_tabla(filas, titulo or leyenda, f"tab:{n}"), "LaTeX"),
            ])
            ediciones.append((m.end(), m.end(),
                              f"<div class='exp'>{tr('Exportar')}: {enlaces}</div>"))

    # ── Figuras (solo las dibujadas con matplotlib, que estan en el registro) ──
    for m in _RE_IMG.finditer(html):
        fig = registro.get(_clave(m.group(1)))
        if fig is None:
            continue
        n = len(figuras) + 1
        # Una figura sin titulo propio (la de calibracion, por ejemplo) toma el
        # del encabezado bajo el que cae, igual que las tablas.
        if not fig.titulo:
            k = bisect.bisect_right(pos_enc, m.start()) - 1
            fig.titulo = encabezados[k][1] if k >= 0 else ""
        cap = _RE_CAPTION.match(html, m.end())
        if cap:
            texto = cap.group(1)
            nuevo = re.sub(r"^(\s*)(Figura|Figure)\.", rf"\g<1>{etq_fig} {n}.", texto, count=1)
            if nuevo == texto:
                nuevo = f"{etq_fig} {n}. {texto.strip()}"
            leyenda = re.sub(r"<[^>]+>", "", nuevo).strip()
            ediciones.append((cap.start(1), cap.end(1), nuevo))
            punto = cap.end()
        else:
            leyenda = f"{etq_fig} {n}. {fig.titulo}".strip()
            ediciones.append((m.end(), m.end(),
                              f"<div class='caption'>{_html.escape(leyenda)}</div>"))
            punto = m.end()
        item = _ItemFigura(n, leyenda, fig, _nombre(etq_fig, n, fig.titulo))
        figuras.append(item)
        if exportar and fig.svg:
            ediciones.append((punto, punto,
                              f"<div class='exp'>{tr('Exportar')}: "
                              + _enlace(item.nombre + ".svg", "image/svg+xml", fig.svg, "SVG")
                              + "</div>"))

    # Se aplican de atras hacia adelante para que las posiciones no se muevan.
    # A igual inicio, la insercion que se anoto primero queda primero.
    resultado = html
    orden = sorted(range(len(ediciones)), key=lambda i: (ediciones[i][0], i), reverse=True)
    for i in orden:
        ini, fin, txt = ediciones[i]
        resultado = resultado[:ini] + txt + resultado[fin:]

    if exportar and (tablas or figuras):
        escribir_carpeta(Path(ruta_informe), tablas, figuras)
    return resultado


def carpeta_de(ruta_informe: Path) -> Path:
    ruta_informe = Path(ruta_informe)
    return ruta_informe.with_name(ruta_informe.stem + "_exportacion")


def escribir_carpeta(ruta_informe: Path, tablas: List[_ItemTabla],
                     figuras: List[_ItemFigura]) -> Path:
    """Escribe tablas/, figuras/, datos/ e indice.csv junto al informe."""
    raiz = carpeta_de(ruta_informe)
    # Solo se borra lo que este modulo crea: si el usuario dejó algo suyo en la
    # carpeta, se respeta. Regenerar el informe no debe dejar figuras viejas
    # mezcladas con las nuevas.
    for sub in (SUBCARPETA_TABLAS, SUBCARPETA_FIGURAS, SUBCARPETA_DATOS):
        shutil.rmtree(raiz / sub, ignore_errors=True)
    (raiz / SUBCARPETA_TABLAS).mkdir(parents=True, exist_ok=True)
    (raiz / SUBCARPETA_FIGURAS).mkdir(parents=True, exist_ok=True)
    (raiz / SUBCARPETA_DATOS).mkdir(parents=True, exist_ok=True)

    etq_tabla, etq_fig = tr("Tabla"), tr("Figura")
    indice: List[list] = []

    for t in tablas:
        d = raiz / SUBCARPETA_TABLAS
        leyenda = f"{etq_tabla} {t.n}. {t.titulo}".strip()
        (d / f"{t.nombre}.csv").write_bytes(_csv_bytes(t.filas[0], t.filas[1:]))
        (d / f"{t.nombre}.tex").write_bytes(_latex_tabla(t.filas, t.titulo or leyenda, f"tab:{t.n}"))
        indice.append([etq_tabla, t.n, f"{SUBCARPETA_TABLAS}/{t.nombre}.csv, .tex", leyenda])
    if tablas:
        (raiz / SUBCARPETA_TABLAS / f"{etq_tabla}s_{idioma().upper()}.xlsx").write_bytes(
            _xlsx_bytes([(f"{etq_tabla} {t.n}", t.filas) for t in tablas]))

    for f in figuras:
        d = raiz / SUBCARPETA_FIGURAS
        if f.fig.svg:
            (d / f"{f.nombre}.svg").write_bytes(f.fig.svg)
            (d / f"{f.nombre}.pdf").write_bytes(f.fig.pdf)
            (d / f"{f.nombre}.png").write_bytes(f.fig.png)
        archivos = [f"{SUBCARPETA_FIGURAS}/{f.nombre}.svg, .pdf, .png"]
        paneles = f.fig.paneles
        for i, (nom, cols, filas) in enumerate(paneles, start=1):
            sufijo = f"_p{i}" if len(paneles) > 1 else ""
            destino = raiz / SUBCARPETA_DATOS / f"{f.nombre}_datos{sufijo}.csv"
            destino.write_bytes(_csv_bytes(cols, filas))
            archivos.append(f"{SUBCARPETA_DATOS}/{destino.name}")
        indice.append([etq_fig, f.n, ", ".join(archivos), f.leyenda])

    (raiz / "indice.csv").write_bytes(
        _csv_bytes([tr("Tipo"), tr("Número"), tr("Archivos"), tr("Leyenda")], indice))
    return raiz
