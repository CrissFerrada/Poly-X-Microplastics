"""Generador del informe de detección en HTML (autocontenido, imágenes en base64).

Contenido:
  1. Abstract (resumen ejecutivo)
  2. Métodos (modelo, params, calibración, dispositivo)
  3. Resultados generales (clase, conf, tamaño, P/R/F1)
  4. Resumen por modelo
  5. Análisis de errores (matriz de confusión + galería)
  6. Comparación entre modelos
  7. Galería completa por imagen anotada
  8. Conteo por muestra y tipo de plástico
  9. Referencias bibliográficas
"""
from __future__ import annotations
import base64
import io
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# matplotlib en modo no-interactivo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import theme as T
from .metrics import (
    aggregate_per_class, confusion_matrix,
    LABEL_TP, LABEL_FP, LABEL_FN, LABEL_MISCLS,
)
from .yolo_wrap import Detection


# ────────────────────────────────────────────────────────────────────
# Estilos del reporte (idénticos al manual)
# ────────────────────────────────────────────────────────────────────
REPORT_CSS = """
:root{--ink:#1f2328;--ink2:#424a53;--ink3:#656d76;--muted:#8c959f;
--rule:#d0d7de;--rule_soft:#eaeef2;--bg:#ffffff;--bg_soft:#f6f8fa;
--accent:#0969da;--accent_d:#0550ae;
--ok:#1f6b5e;--warn:#9a6700;--err:#cf222e;--vio:#6639ba;}
*{box-sizing:border-box;}
body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:var(--ink);
background:var(--bg);margin:0;padding:0;line-height:1.55;}
.container{max-width:1100px;margin:0 auto;padding:40px 48px 80px;}
header.cover{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:28px;}
.kicker{font-size:10pt;letter-spacing:.14em;text-transform:uppercase;
color:var(--accent_d);font-weight:600;margin-bottom:8px;}
h1{font-size:28pt;margin:4px 0 8px;line-height:1.18;}
h2{font-size:18pt;margin-top:38px;padding-bottom:8px;border-bottom:1px solid var(--rule);}
h3{font-size:13pt;margin-top:22px;color:var(--ink2);}
h4{font-size:11pt;margin-top:14px;color:var(--ink2);}
.meta{font-size:10pt;color:var(--ink3);}
.toc{background:var(--bg_soft);border-left:3px solid var(--accent);
padding:14px 20px;margin:18px 0 28px;}
.toc h3{margin:0 0 8px;color:var(--ink);}
.toc ol{margin:0;padding-left:22px;font-size:10.5pt;}
.toc a{color:var(--ink2);text-decoration:none;}
.toc a:hover{color:var(--accent);}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0;}
.kpi{background:var(--bg);border:1px solid var(--rule);border-radius:8px;
padding:14px 16px;}
.kpi .v{font-size:22pt;font-weight:700;color:var(--ink);margin:4px 0 6px;}
.kpi .l{font-size:9pt;font-weight:600;color:var(--ink3);letter-spacing:1.4px;
text-transform:uppercase;}
.kpi .b{height:3px;background:var(--accent);border-radius:2px;}
table.data{width:100%;border-collapse:collapse;font-size:10pt;margin:12px 0;}
table.data th{background:var(--bg_soft);text-align:left;padding:8px 10px;
border-bottom:1px solid var(--rule);font-weight:600;}
table.data td{padding:7px 10px;border-bottom:1px solid var(--rule_soft);}
table.data tr:last-child td{border-bottom:none;}
table.data td.r{text-align:right;font-variant-numeric:tabular-nums;}
.fig{margin:18px 0;}
.fig img{width:100%;border:1px solid var(--rule);border-radius:6px;display:block;}
.caption{font-size:9pt;color:var(--ink3);margin-top:6px;text-align:center;}
.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0;}
.gallery .item{border:1px solid var(--rule);border-radius:6px;overflow:hidden;background:var(--bg_soft);}
.gallery .item img{width:100%;display:block;}
.gallery .item .cap{font-size:8.5pt;color:var(--ink3);padding:5px 8px;}
/* Comparación lado a lado: Predicción vs Ground Truth */
.compare{border:1px solid var(--rule);border-radius:8px;overflow:hidden;
margin:18px 0;background:var(--bg_soft);}
.compare-pair{display:grid;grid-template-columns:1fr 1fr;gap:0;}
.compare-pair figure{margin:0;border-right:1px solid var(--rule);background:var(--bg);}
.compare-pair figure:last-child{border-right:none;}
.compare-pair img{width:100%;display:block;}
.compare-pair figcaption{font-size:9pt;color:var(--ink2);font-weight:600;
padding:7px 10px;text-align:center;border-top:1px solid var(--rule_soft);}
.compare-pair figcaption .sub{display:block;font-weight:400;color:var(--ink3);font-size:8.5pt;margin-top:2px;}
.compare-meta{font-size:9pt;color:var(--ink2);padding:9px 12px;
border-top:1px solid var(--rule);font-variant-numeric:tabular-nums;}
.compare-meta .tag{display:inline-block;margin-right:10px;}
/* Conteo por polimero de una sola foto, bajo el par de imagenes. */
table.conteo{width:100%;border-collapse:collapse;font-size:9.5pt;
border-top:1px solid var(--rule);background:var(--bg);}
table.conteo th{background:var(--bg_soft);text-align:right;padding:6px 12px;
font-weight:600;color:var(--ink2);border-bottom:1px solid var(--rule_soft);}
table.conteo th:first-child{text-align:left;}
table.conteo td{padding:5px 12px;text-align:right;
font-variant-numeric:tabular-nums;border-bottom:1px solid var(--rule_soft);}
table.conteo td:first-child{text-align:left;font-weight:600;}
table.conteo tr.tot td{border-bottom:none;background:var(--bg_soft);font-weight:700;}
table.conteo td.dif-pos{color:var(--err);}
table.conteo td.dif-neg{color:var(--accent_d);}
.nogt{display:flex;align-items:center;justify-content:center;min-height:180px;
color:var(--muted);font-size:10pt;text-align:center;padding:16px;
background:repeating-linear-gradient(45deg,#fbfcfd,#fbfcfd 12px,#f0f2f4 12px,#f0f2f4 24px);}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:8.5pt;
font-weight:600;letter-spacing:.04em;}
.b-blue{background:#ddf4ff;color:var(--accent_d);}
.b-green{background:#d8f5ea;color:var(--ok);}
.b-amber{background:#fff8c5;color:var(--warn);}
.b-red{background:#ffebe9;color:var(--err);}
.b-violet{background:#ece0fd;color:var(--vio);}
footer{border-top:1px solid var(--rule);margin-top:60px;padding-top:18px;
font-size:9.5pt;color:var(--ink3);}
@media print{.container{max-width:none;padding:24px;}
h2{page-break-after:avoid;}.fig,.gallery .item{page-break-inside:avoid;}
/* Que la foto y su tabla no queden en paginas distintas del PDF. */
.compare{page-break-inside:avoid;}}
"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ────────────────────────────────────────────────────────────────────
# Conteo por polímero y por muestra
# ────────────────────────────────────────────────────────────────────
# Orden de presentación de los polímeros. Cualquier clase que aparezca en los
# datos y no esté aquí se agrega al final, para que un dataset con otras clases
# no pierda columnas en silencio.
_ORDEN_CLASES = ["PET", "PP", "LDPE"]

# Las placas del río Loa se nombran `<tramo>.<testigo>` con sufijo `x`/`xx` para
# la segunda y tercera placa del mismo tramo. Los recortes agregan `__f<fila>c<col>`.
_RE_PLACA = re.compile(r"^(\d+)\.(\d+)(x*)$")
_ESTACION = {"1": "Chiu Chiu", "2": "Taira", "3": "Desembocadura"}
_ORDEN_ESTACION = ["Chiu Chiu", "Taira", "Desembocadura"]
_PLACA = {"": "a", "x": "b", "xx": "c"}


def _conteo_por_clase(detecciones) -> Counter:
    c = Counter()
    for d in detecciones:
        c[d.class_name] += 1
    return c


def _clases_vistas(resultados_planos) -> List[str]:
    """Clases presentes en el lote, en orden de presentación.

    Se toman de las detecciones **y** del Ground Truth: un polímero que el
    modelo nunca detecta seguiría siendo una fila de la tabla, con cero, que es
    justo el dato que interesa mirar.
    """
    vistas = {d.class_name for r in resultados_planos
              for d in list(r.predictions) + list(r.gt)}
    orden = [c for c in _ORDEN_CLASES if c in vistas]
    return orden + sorted(vistas - set(orden))


def _muestra_de(ruta) -> Optional[tuple]:
    """(estación, tramo, placa) a partir del nombre, o None si no sigue la pauta.

    El informe no puede exigir esta nomenclatura: si las imágenes vienen de otro
    estudio, la tabla agrupada simplemente no se emite y queda la tabla por
    imagen, que siempre se puede construir.
    """
    stem = Path(ruta).stem.split("__")[0]
    m = _RE_PLACA.match(stem)
    if not m:
        return None
    tramo, testigo, sufijo = m.group(1), m.group(2), m.group(3)
    return (_ESTACION.get(testigo, f"Testigo {testigo}"),
            int(tramo), _PLACA.get(sufijo, sufijo))


def _clave_orden(ruta) -> tuple:
    """Ordena por estación/tramo/placa cuando el nombre lo permite."""
    m = _muestra_de(ruta)
    if m is None:
        return (1, 0, 0, "", Path(ruta).name)
    estacion, tramo, placa = m
    idx = _ORDEN_ESTACION.index(estacion) if estacion in _ORDEN_ESTACION else 99
    return (0, idx, tramo, placa, Path(ruta).name)


def _tabla_conteo(manual: Optional[Counter], detectado: Counter,
                  clases: List[str]) -> str:
    """Tabla de una foto: partículas por polímero, contadas a mano y por el modelo.

    Es conteo, no evaluación: la columna del modelo son **todas** sus
    detecciones de ese polímero, sin descontar falsos positivos ni emparejar
    caja a caja. Para eso está el análisis de errores.
    """
    hay_manual = manual is not None
    filas = ""
    for c in clases:
        det = detectado.get(c, 0)
        if hay_manual:
            man = manual.get(c, 0)
            dif = det - man
            css = "dif-pos" if dif > 0 else ("dif-neg" if dif < 0 else "")
            filas += (f"<tr><td>{c}</td><td>{man}</td><td>{det}</td>"
                      f"<td class='{css}'>{dif:+d}</td></tr>")
        else:
            filas += f"<tr><td>{c}</td><td>{det}</td></tr>"

    tot_det = sum(detectado.get(c, 0) for c in clases)
    if hay_manual:
        tot_man = sum(manual.get(c, 0) for c in clases)
        dif = tot_det - tot_man
        css = "dif-pos" if dif > 0 else ("dif-neg" if dif < 0 else "")
        filas += (f"<tr class='tot'><td>Total</td><td>{tot_man}</td>"
                  f"<td>{tot_det}</td><td class='{css}'>{dif:+d}</td></tr>")
        cabecera = ("<tr><th>Polímero</th><th>Conteo manual</th>"
                    "<th>Detectadas por el modelo</th><th>Diferencia</th></tr>")
    else:
        filas += f"<tr class='tot'><td>Total</td><td>{tot_det}</td></tr>"
        cabecera = "<tr><th>Polímero</th><th>Detectadas por el modelo</th></tr>"

    return f"<table class='conteo'>{cabecera}{filas}</table>"


# Ancho máximo de las imágenes de galería, en píxeles. Se incrustan en base64
# dentro del HTML, así que su tamaño es el del reporte: sin reducir, cuatro
# imágenes de 4096 px pesaban 14.5 MB y un lote de varios cientos generaba un
# archivo de gigabytes que ningún navegador abre.
_ANCHO_GALERIA = 1100


def _uri_galeria(data: bytes, ancho_max: int = _ANCHO_GALERIA) -> str:
    """Data URI de una imagen de galería: siempre JPEG, reescalada si es ancha.

    Se recodifica **siempre**, no solo al reducir. Las anotadas se generan en
    PNG (sin pérdida), formato pésimo para fotografía: una sola placa ocupaba
    ~3.6 MB y cuatro imágenes producían un reporte de 14.5 MB. En JPEG la misma
    imagen baja un orden de magnitud sin diferencia visible.

    Devuelve el URI completo con su tipo MIME, no solo el base64: declararlo
    como PNG tras recodificar impediría que el navegador lo mostrara.
    """
    try:
        import cv2
        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            if arr.shape[1] > ancho_max:
                alto = int(round(arr.shape[0] * ancho_max / arr.shape[1]))
                arr = cv2.resize(arr, (ancho_max, alto), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                return ("data:image/jpeg;base64,"
                        + base64.b64encode(buf.tobytes()).decode("ascii"))
    except Exception:
        pass
    return "data:image/png;base64," + _img_b64(data)


def _equipo() -> List[tuple]:
    """Componentes de la maquina, para poder declarar donde se ejecuto."""
    import platform

    filas = [("Sistema operativo", f"{platform.system()} {platform.release()}"),
             ("Procesador", platform.processor() or "—")]
    try:
        import psutil
        filas.append(("Memoria RAM",
                      f"{psutil.virtual_memory().total / 1024**3:.1f} GB"))
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            filas += [
                ("GPU", props.name),
                ("VRAM", f"{props.total_memory / 1024**3:.1f} GB"),
                ("Compute capability", f"{props.major}.{props.minor}"),
                ("PyTorch", f"{torch.__version__} (CUDA {torch.version.cuda})"),
            ]
        else:
            filas += [("GPU", "no disponible"), ("PyTorch", torch.__version__)]
    except Exception:
        pass
    return filas


def _entrenamiento_de(peso: Path) -> Optional[Dict]:
    """Lee del checkpoint con que configuracion y resultado se entreno.

    Ultralytics guarda ``train_args`` y ``train_metrics`` dentro del propio .pt,
    asi que el informe puede declarar imgsz, batch, epocas y metricas sin
    depender de que la carpeta del run siga existiendo.
    """
    try:
        import torch
        ck = torch.load(str(peso), map_location="cpu", weights_only=False)
    except Exception:
        return None
    args = ck.get("train_args") or {}
    met = ck.get("train_metrics") or {}
    if not args and not met:
        return None
    return {
        "base": args.get("model", "—"),
        "imgsz": args.get("imgsz", "—"),
        "batch": args.get("batch", "—"),
        "epocas": args.get("epochs", "—"),
        "optimizador": args.get("optimizer", "—"),
        "lr0": args.get("lr0", "—"),
        "amp": args.get("amp", "—"),
        "precision": met.get("metrics/precision(B)"),
        "recall": met.get("metrics/recall(B)"),
        "map50": met.get("metrics/mAP50(B)"),
        "map": met.get("metrics/mAP50-95(B)"),
    }


def _seccion_equipo(active) -> str:
    """Tabla de componentes y de como se entreno cada modelo cargado."""
    filas_eq = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in _equipo())
    html = ("<h3>@@N@@.1 Equipo de cómputo</h3>"
            f"<table class='data'><tr><th>Componente</th><th>Detalle</th></tr>"
            f"{filas_eq}</table>")

    entrenos = []
    for s in active:
        if s.path is None:
            continue
        info = _entrenamiento_de(Path(s.path))
        if info:
            entrenos.append((s.alias, info))
    if not entrenos:
        return html

    def _n(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    filas = ""
    for alias, i in entrenos:
        filas += (
            f"<tr><td>{alias}</td><td>{i['base']}</td>"
            f"<td class='r'>{i['imgsz']}</td><td class='r'>{i['batch']}</td>"
            f"<td class='r'>{i['epocas']}</td><td>{i['optimizador']}</td>"
            f"<td class='r'>{_n(i['precision'])}</td><td class='r'>{_n(i['recall'])}</td>"
            f"<td class='r'>{_n(i['map50'])}</td><td class='r'>{_n(i['map'])}</td></tr>")

    html += (
        "<h3>@@N@@.2 Entrenamiento de cada modelo</h3>"
        "<p>Configuración y métricas de validación con que se entrenó cada peso, "
        "leídas del propio archivo <code>.pt</code>.</p>"
        "<table class='data'>"
        "<tr><th>Modelo</th><th>Arquitectura base</th><th>imgsz</th><th>batch</th>"
        "<th>épocas</th><th>optimizador</th><th>Precisión</th><th>Recall</th>"
        "<th>mAP@50</th><th>mAP@50-95</th></tr>"
        f"{filas}</table>")
    return html


def _barrido_confianza(resultados, state, iou_thr: float, conf_actual: float):
    """Recalcula P/R/F1 a distintos umbrales de confianza, sin re-inferir.

    Las predicciones guardadas ya pasaron el umbral con que se ejecuto, asi que
    subirlo es solo filtrar y volver a emparejar. Bajarlo NO se puede: esas
    detecciones nunca se calcularon. Por eso el barrido arranca en el umbral
    usado y avisa de esa limitacion.

    Devuelve {alias: [(conf, precision, recall, f1, n_det), ...]}.
    """
    from .metrics import match_image

    umbrales = [round(u, 2) for u in np.arange(conf_actual, 0.96, 0.05)]
    salida = {}
    for mi, slot in enumerate(state.model_slots):
        if slot.path is None:
            continue
        rs = [r for r in resultados.get(mi, []) if r.has_gt]
        if not rs:
            continue
        filas = []
        for u in umbrales:
            tp = fp = fn = mc = nd = 0
            for r in rs:
                preds = [p for p in r.predictions if p.conf >= u]
                nd += len(preds)
                m = match_image(preds, r.gt, iou_thr=iou_thr)
                tp += m.tp; fp += m.fp; fn += m.fn; mc += m.miscls
            # Estricto: la mala clasificacion penaliza en ambos lados.
            p = tp / (tp + fp + mc) if (tp + fp + mc) else 0.0
            rec = tp / (tp + fn + mc) if (tp + fn + mc) else 0.0
            f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
            filas.append((u, p, rec, f1, nd))
        salida[slot.alias] = filas
    return salida


def _fig_barrido(barrido: Dict[str, list]) -> str:
    """Curva de F1 frente al umbral de confianza, una linea por modelo."""
    if not barrido:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for alias, filas in barrido.items():
        ax.plot([f[0] for f in filas], [f[3] for f in filas],
                marker="o", markersize=3, label=alias, linewidth=1.6)
        mejor = max(filas, key=lambda f: f[3])
        ax.plot([mejor[0]], [mejor[3]], marker="*", markersize=13, zorder=5,
                color=ax.lines[-1].get_color())
    ax.set_xlabel("Umbral de confianza")
    ax.set_ylabel("F1 (con clase)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


_PALETA = (T.ACCENT, T.VIO, T.WARN)


def _fig_comparacion_metricas(datos: List[tuple]) -> str:
    """Barras agrupadas de P/R/F1 por modelo, en los dos criterios.

    ``datos``: ``[(alias, p_loc, r_loc, f1_loc, p_cls, r_cls, f1_cls), ...]``.

    Se dibujan los dos paneles juntos y no solo el permisivo: la distancia entre
    ambos es la confusion entre polimeros, y verla al lado evita leer el numero
    bueno como si fuera el desempeno completo.
    """
    if not datos:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    etiquetas = ["Precisión", "Recall", "F1"]
    x = np.arange(len(etiquetas))
    ancho = 0.8 / max(1, len(datos))
    for panel, (ax, desplazamiento, titulo) in enumerate((
            (axes[0], 1, "Localización"), (axes[1], 4, "Con clase"))):
        for i, fila in enumerate(datos):
            vals = fila[desplazamiento:desplazamiento + 3]
            pos = x - 0.4 + ancho * (i + 0.5)
            barras = ax.bar(pos, vals, ancho, label=fila[0],
                            color=_PALETA[i % len(_PALETA)])
            if panel == 1 or len(datos) <= 2:
                ax.bar_label(barras, fmt="%.3f", fontsize=7, padding=1)
        ax.set_title(titulo, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Valor")
    # La leyenda va bajo la figura: dentro de los ejes se montaba sobre las
    # barras, que con estos valores llegan casi al techo en las dos mitades.
    manejadores, nombres = axes[0].get_legend_handles_labels()
    fig.legend(manejadores, nombres, frameon=False, fontsize=9,
               loc="lower center", ncol=len(datos),
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_acuerdo_por_imagen(alias_a: str, alias_b: str, pares: List[tuple]) -> str:
    """Detecciones por imagen de un modelo frente al otro, con la diagonal.

    El total de un lote esta dominado por unas pocas placas densas, asi que dos
    modelos pueden empatar en la suma y discrepar foto a foto. Aqui cada punto es
    una imagen: sobre la diagonal cuentan igual, y la distancia a ella dice
    cuanto se separan y en que placas.

    Escala simetrica-logaritmica porque la mayoria de las placas tiene 0-10
    particulas y unas pocas pasan de 500; en escala lineal el grueso de los
    puntos se apelotona en el origen.
    """
    if not pares:
        return ""
    a = [p[0] for p in pares]
    b = [p[1] for p in pares]
    tope = max(max(a), max(b), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, tope], [0, tope], "--", color="#999", linewidth=1,
            label="acuerdo exacto")
    ax.scatter(a, b, s=26, alpha=0.75, color=T.ACCENT, edgecolor="white",
               linewidth=0.5)
    # Se rotulan las placas que mas se separan: son las que hay que ir a mirar.
    for na, nb, nombre in sorted(pares, key=lambda p: -abs(p[0] - p[1]))[:3]:
        if abs(na - nb) > 0:
            ax.annotate(nombre, (na, nb), fontsize=7, color=T.INK2,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlim(-1, tope * 1.6)
    ax.set_ylim(-1, tope * 1.6)
    ax.set_xlabel(f"Detecciones · {alias_a}")
    ax.set_ylabel(f"Detecciones · {alias_b}")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_metricas_por_clase(por_modelo: Dict[str, Dict[str, tuple]]) -> str:
    """F1 por clase y por modelo: donde gana cada uno.

    El F1 global promedia polimeros con desempeno muy distinto -- PET separa
    bien, PP y LDPE comparten tono y se confunden -- de modo que un modelo puede
    ganar en el agregado y perder en la clase que interesa.
    """
    if not por_modelo:
        return ""
    clases = []
    for m in por_modelo.values():
        for c in m:
            if c not in clases:
                clases.append(c)
    if not clases:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(clases))
    ancho = 0.8 / max(1, len(por_modelo))
    for i, (alias, metricas) in enumerate(por_modelo.items()):
        vals = [metricas.get(c, (0, 0, 0))[2] for c in clases]
        barras = ax.bar(x - 0.4 + ancho * (i + 0.5), vals, ancho, label=alias,
                        color=_PALETA[i % len(_PALETA)])
        ax.bar_label(barras, fmt="%.3f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(clases)
    ax.set_ylabel("F1 por clase")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_class_distribution(per_class_counts: Dict[str, int]) -> str:
    if not per_class_counts:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = list(per_class_counts.keys())
    vals = list(per_class_counts.values())
    colors = [T.CLASS_COLOR_HEX.get(n, "#888") for n in names]
    ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Detecciones")
    ax.set_title("Distribución por clase")
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _fig_confidence_hist(confs: List[float]) -> str:
    if not confs:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(confs, bins=20, color=T.ACCENT, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Confianza")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Histograma de confianza")
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _fig_size_hist(sizes_um: List[float]) -> str:
    if not sizes_um:
        return ""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(sizes_um, bins=25, color=T.WARN, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Diámetro equivalente (μm)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de tamaños")
    ax.grid(axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _fig_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> str:
    labels = list(class_names) + ["background"]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Ground Truth")
    ax.set_title("Matriz de confusión")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() * 0.5 else "black"
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=9, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_b64(fig)


def _tabla_ancha(filas: List[tuple], clases: List[str], titulo_col: str,
                hay_manual: bool, totales: bool = True) -> str:
    """Tabla Muestra x polimero con las columnas manual y modelo lado a lado.

    ``filas`` es una lista de (etiqueta, Counter manual, Counter modelo).
    """
    if hay_manual:
        cab = (f"<tr><th rowspan='2'>{titulo_col}</th>"
               + "".join(f"<th colspan='2' style='text-align:center'>{c}</th>"
                         for c in clases)
               + "<th colspan='2' style='text-align:center'>Total</th></tr><tr>"
               + "".join("<th class='r'>manual</th><th class='r'>modelo</th>"
                         for _ in clases + ["total"]) + "</tr>")
    else:
        cab = (f"<tr><th>{titulo_col}</th>"
               + "".join(f"<th class='r'>{c}</th>" for c in clases)
               + "<th class='r'>Total</th></tr>")

    cuerpo = ""
    acum_man, acum_det = Counter(), Counter()
    for etiqueta, man, det in filas:
        acum_man.update(man)
        acum_det.update(det)
        celdas = ""
        for c in clases:
            if hay_manual:
                celdas += f"<td class='r'>{man.get(c, 0)}</td><td class='r'>{det.get(c, 0)}</td>"
            else:
                celdas += f"<td class='r'>{det.get(c, 0)}</td>"
        t_det = sum(det.get(c, 0) for c in clases)
        if hay_manual:
            t_man = sum(man.get(c, 0) for c in clases)
            celdas += f"<td class='r'><strong>{t_man}</strong></td><td class='r'><strong>{t_det}</strong></td>"
        else:
            celdas += f"<td class='r'><strong>{t_det}</strong></td>"
        cuerpo += f"<tr><td>{etiqueta}</td>{celdas}</tr>"

    if totales and filas:
        celdas = ""
        for c in clases:
            if hay_manual:
                celdas += (f"<td class='r'>{acum_man.get(c, 0)}</td>"
                           f"<td class='r'>{acum_det.get(c, 0)}</td>")
            else:
                celdas += f"<td class='r'>{acum_det.get(c, 0)}</td>"
        t_det = sum(acum_det.get(c, 0) for c in clases)
        if hay_manual:
            t_man = sum(acum_man.get(c, 0) for c in clases)
            celdas += f"<td class='r'>{t_man}</td><td class='r'>{t_det}</td>"
        else:
            celdas += f"<td class='r'>{t_det}</td>"
        cuerpo += (f"<tr style='font-weight:700;background:var(--bg_soft)'>"
                   f"<td>TOTAL</td>{celdas}</tr>")

    return f"<table class='data'>{cab}{cuerpo}</table>"


def _seccion_conteo(resultados: Dict[int, list], state, active,
                    clases: List[str]) -> str:
    """Seccion 8: cuantas particulas de cada polimero hay en cada muestra.

    Es la tabla que se pide para el manuscrito: conteo por tipo de plastico y
    muestra, con el conteo manual y el del modelo uno junto al otro. **No** es
    evaluacion del detector: la columna del modelo son todas sus detecciones,
    sin descontar falsos positivos. Esa lectura esta en la seccion de errores.

    Con varios modelos cargados se usa el primero activo. Poner dos modelos en
    la misma celda haria la tabla ilegible, y el informe ya compara modelos en
    su propia seccion.
    """
    if not active:
        return ""
    main_mi = state.model_slots.index(active[0])
    rs = resultados.get(main_mi, [])
    if not rs:
        return ""

    rs = sorted(rs, key=lambda r: _clave_orden(r.image_path))
    hay_manual = any(r.has_gt for r in rs)

    # -- Por imagen --
    filas_img = []
    for r in rs:
        muestra = _muestra_de(r.image_path)
        etiqueta = r.image_path.name
        if muestra:
            estacion, tramo, placa = muestra
            etiqueta = (f"{r.image_path.name}<br>"
                        f"<span style='font-size:8.5pt;color:var(--ink3)'>"
                        f"{estacion} &middot; tramo {tramo} &middot; placa {placa}</span>")
        filas_img.append((etiqueta,
                          _conteo_por_clase(r.gt) if r.has_gt else Counter(),
                          _conteo_por_clase(r.predictions)))
    tabla_img = _tabla_ancha(filas_img, clases, "Muestra (imagen)", hay_manual)

    # -- Por tramo y por estacion, si los nombres siguen la pauta --
    agrupadas = [(r, _muestra_de(r.image_path)) for r in rs]
    parseables = [(r, m) for r, m in agrupadas if m is not None]
    tabla_tramo = tabla_estacion = ""
    nota_agrupacion = ""
    if parseables:
        por_tramo: Dict[tuple, list] = {}
        por_estacion: Dict[str, list] = {}
        for r, (estacion, tramo, _placa) in parseables:
            por_tramo.setdefault((estacion, tramo), []).append(r)
            por_estacion.setdefault(estacion, []).append(r)

        def _sumar(lista):
            man, det = Counter(), Counter()
            for r in lista:
                if r.has_gt:
                    man.update(_conteo_por_clase(r.gt))
                det.update(_conteo_por_clase(r.predictions))
            return man, det

        def _orden_est(e):
            return _ORDEN_ESTACION.index(e) if e in _ORDEN_ESTACION else 99

        filas_t = []
        for clave in sorted(por_tramo, key=lambda k: (_orden_est(k[0]), k[1])):
            estacion, tramo = clave
            lista = por_tramo[clave]
            man, det = _sumar(lista)
            palabra = "imágenes" if len(lista) != 1 else "imagen"
            filas_t.append((f"{estacion} &middot; tramo {tramo} "
                            f"<span style='font-size:8.5pt;color:var(--ink3)'>"
                            f"({len(lista)} {palabra})</span>",
                            man, det))
        tabla_tramo = ("<h3>@@N@@.2 Por tramo de profundidad</h3>"
                       "<p>Las placas del mismo tramo se <strong>suman</strong>: son "
                       "submuestras de la misma masa de sedimento, no repeticiones "
                       "fotográficas. El tramo es la unidad de análisis.</p>"
                       + _tabla_ancha(filas_t, clases, "Tramo", hay_manual))

        filas_e = []
        for estacion in sorted(por_estacion, key=_orden_est):
            man, det = _sumar(por_estacion[estacion])
            filas_e.append((estacion, man, det))
        tabla_estacion = ("<h3>@@N@@.3 Por estación</h3>"
                          + _tabla_ancha(filas_e, clases, "Estación", hay_manual,
                                         totales=False))

        fuera = len(agrupadas) - len(parseables)
        if fuera:
            nota_agrupacion = (f"<p class='caption'>{fuera} imagen(es) no siguen la "
                               f"nomenclatura <code>tramo.testigo</code> y quedan fuera "
                               f"de las tablas agrupadas; sí están en la tabla por "
                               f"imagen.</p>")

    coletilla = ", el primer modelo activo." if len(active) > 1 else "."
    intro = (
        ""
        "<p>Part&iacute;culas contadas en cada muestra, desglosadas por pol&iacute;mero. "
        "La columna <em>manual</em> es la anotaci&oacute;n humana (Ground Truth) y la "
        "columna <em>modelo</em> son todas las detecciones de "
        f"<strong>{active[0].alias}</strong>{coletilla}</p>"
        "<p><strong>L&eacute;ase como conteo, no como evaluaci&oacute;n.</strong> La "
        "columna del modelo no descuenta falsos positivos ni empareja caja a caja: es "
        "cu&aacute;ntas part&iacute;culas de cada pol&iacute;mero report&oacute;. "
        "Coincidir en el total no implica haber acertado part&iacute;cula por "
        "part&iacute;cula; para eso est&aacute; el an&aacute;lisis de errores.</p>"
        "<h3>@@N@@.1 Por imagen</h3>"
    )
    return intro + tabla_img + nota_agrupacion + tabla_tramo + tabla_estacion


# ────────────────────────────────────────────────────────────────────
# ── Secciones del informe ────────────────────────────────────────────────────
# El orden de esta lista es el orden del informe, y la numeracion sale de aqui:
# si se desmarca una seccion las siguientes se renumeran solas, en vez de dejar
# un hueco ("4, 6, 7") que en un documento entregable se lee como un error.
#
# Anadir una seccion nueva es anadir una tupla: la interfaz dibuja sus casillas
# leyendo esta lista, de modo que no hay una segunda lista que mantener en
# sincronia.
SECCIONES = [
    ("resumen",     "Resumen"),
    ("metodos",     "Métodos"),
    ("calibracion", "Calibración de escala"),
    ("forma",       "Forma y talla de las partículas"),
    ("resultados",  "Resultados generales"),
    ("modelos",     "Resumen por modelo"),
    ("errores",     "Análisis de errores"),
    ("comparacion", "Comparación entre modelos"),
    ("galeria",     "Galería por imagen"),
    ("conteo",      "Conteo por muestra y tipo de plástico"),
    ("referencias", "Referencias bibliográficas"),
]

IDS_SECCIONES = [s[0] for s in SECCIONES]

# Combinaciones de uso frecuente. Existen porque el problema real no era que no
# se pudiera elegir, sino que elegir diez casillas una por una es tedioso y
# nadie lo hace: con un preset el informe corto queda a un clic.
PRESETS = {
    "completo": IDS_SECCIONES,
    "resumen": ["resumen", "calibracion", "forma", "conteo", "galeria"],
    "metodologico": ["resumen", "metodos", "calibracion", "forma", "modelos",
                     "errores", "comparacion", "referencias"],
}

# Marca que se sustituye por el numero definitivo de la seccion al ensamblar.
# Los numeros no se pueden escribir a mano en los titulos porque dependen de que
# otras secciones se hayan pedido.
NUM = "@@N@@"


def generate_report(state, output_path: Path,
                    include_refs: bool = True,
                    include_gallery: bool = True,
                    max_gallery: int = 60,
                    solo_imagenes=None,
                    secciones=None) -> Path:
    """Genera el reporte HTML. `state` es un DetectorState con resultados.

    ``max_gallery`` limita cuántas imágenes se incrustan en la galería. Las
    imágenes van en base64 dentro del propio HTML, de modo que sin tope un lote
    grande generaba un archivo inabrible (14.5 MB con solo 4 imágenes).

    ``solo_imagenes`` acota el informe a un conjunto de rutas. Con ``None`` entra
    el trabajo completo. El filtro se aplica antes de calcular nada, de forma que
    los totales, los gráficos y la matriz de confusión describan exactamente el
    mismo subconjunto que se muestra; si se filtrara solo la galería, las cifras
    de arriba hablarían de un lote y las imágenes de otro.

    ``secciones`` es el conjunto de identificadores de :data:`SECCIONES` que se
    quieren incluir; ``None`` las incluye todas. Una sección pedida que no tenga
    contenido -- errores sin ground truth, por ejemplo -- se omite igualmente:
    marcarla no inventa datos que no existen.
    """
    pedidas = set(IDS_SECCIONES if secciones is None else secciones)
    # Las dos casillas antiguas siguen mandando si se pasan en False, para no
    # cambiar el significado de las llamadas que ya existian.
    if not include_refs:
        pedidas.discard("referencias")
    if not include_gallery:
        pedidas.discard("galeria")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    active = state.active_models()

    if solo_imagenes is None:
        resultados = dict(state.results)
    else:
        elegidas = {Path(p) for p in solo_imagenes}
        resultados = {mi: [r for r in rs if Path(r.image_path) in elegidas]
                      for mi, rs in state.results.items()}

    all_results = [r for rs in resultados.values() for r in rs]
    total_imgs = len({r.image_path for r in all_results})
    total_dets = sum(len(r.predictions) for r in all_results)
    confs = [p.conf for r in all_results for p in r.predictions]
    sizes = [p.diam_um for r in all_results for p in r.predictions if p.diam_um]
    avg_conf = sum(confs)/len(confs) if confs else 0
    avg_size = sum(sizes)/len(sizes) if sizes else 0
    any_gt = any(r.has_gt for r in all_results)

    clases_lote = _clases_vistas(all_results)

    # ── Distribución por clase ──
    per_class = Counter()
    for r in all_results:
        for p in r.predictions:
            per_class[p.class_name] += 1

    # ── Análisis de errores: matriz + métricas por clase (modelo principal) ──
    err_section = ""
    if any_gt and active:
        main_mi = state.model_slots.index(active[0])
        rs = resultados.get(main_mi, [])
        gts_only = [r.gt for r in rs if r.has_gt]
        preds_only = [r.predictions for r in rs if r.has_gt]
        cls_ids = sorted({d.class_id for lst in gts_only + preds_only for d in lst})
        cls_names = []
        for cid in cls_ids:
            name = next((d.class_name for lst in gts_only + preds_only for d in lst if d.class_id == cid), str(cid))
            cls_names.append(name)
        cm = confusion_matrix(preds_only, gts_only, cls_ids, iou_thr=state.params.iou_tp)
        per_class_metrics = aggregate_per_class(preds_only, gts_only, cls_ids, iou_thr=state.params.iou_tp)

        cm_img = _fig_confusion_matrix(cm, cls_names)
        rows = ""
        for cid, name in zip(cls_ids, cls_names):
            cm_ = per_class_metrics[cid]
            rows += (
                f"<tr><td>{name}</td>"
                f"<td class='r'>{cm_.tp}</td><td class='r'>{cm_.fp}</td><td class='r'>{cm_.fn}</td>"
                f"<td class='r'>{cm_.precision:.3f}</td><td class='r'>{cm_.recall:.3f}</td>"
                f"<td class='r'>{cm_.f1:.3f}</td></tr>"
            )
        # Las clases que el modelo conoce pero que no aparecen ni en el GT ni en
        # las predicciones se quedan fuera de la tabla, porque no tienen metrica
        # que calcular. Se declaran igual: un informe que anuncia PET/PP/LDPE y
        # luego muestra dos filas parece roto, y el lector no puede distinguir
        # "ausente del lote" de "se perdio por el camino".
        conocidas = dict(getattr(getattr(active[0], "loaded", None), "names", {}) or {})
        ausentes = [n for cid, n in sorted(conocidas.items()) if cid not in set(cls_ids)]
        nota_ausentes = ""
        if ausentes:
            nota_ausentes = (
                f"<p class='caption' style='text-align:left'>Sin fila para "
                f"{', '.join(ausentes)}: no aparece ni en la anotación manual ni "
                f"entre las predicciones de este lote, de modo que no hay métrica "
                f"que informar. El modelo sí está entrenado para esa clase.</p>")

        err_section = f"""
        
        <h3>@@N@@.1 Matriz de confusión</h3>
        <div class='fig'><img src='data:image/png;base64,{cm_img}' />
            <div class='caption'>Figura. Matriz de confusión (modelo principal: {active[0].alias}, IoU = {state.params.iou_tp}).</div></div>
        <h3>@@N@@.2 Precisión / Recall / F1 por clase</h3>
        <table class='data'><tr><th>Clase</th><th>{LABEL_TP}</th><th>{LABEL_FP}</th><th>{LABEL_FN}</th>
        <th>Precisión</th><th>Recall</th><th>F1</th></tr>{rows}</table>
        {nota_ausentes}
        """

    # ── Resumen por modelo (tabla comparativa) ──
    #
    # Se reportan DOS F1, y la diferencia entre ambos no es un detalle:
    #
    #   localizacion  tp/(tp+fp) y tp/(tp+fn). Las cajas bien situadas pero con
    #                 la clase equivocada quedan fuera de ambos denominadores,
    #                 asi que mide solo si el detector encuentra la particula.
    #   con clase     esas mismas cajas cuentan como falso positivo de la clase
    #                 predicha y falso negativo de la real, que es lo estricto.
    #
    # Publicar solo el primero sobreestima el desempeno, y ademas contradice la
    # tabla por clase, que si penaliza la confusion. Se declaran los dos.
    def _pr_f1(tp: int, fp: int, fn: int):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)

    rows_models = ""
    total_mc = 0
    for mi, slot in enumerate(state.model_slots):
        if slot.path is None: continue
        rs = resultados.get(mi, [])
        n_img = len({r.image_path for r in rs})
        n_det = sum(len(r.predictions) for r in rs)
        cf = [p.conf for r in rs for p in r.predictions]
        tp = sum(r.tp for r in rs); fp = sum(r.fp for r in rs)
        fn = sum(r.fn for r in rs); mc = sum(r.miscls for r in rs)
        total_mc += mc
        _, _, f1_loc = _pr_f1(tp, fp, fn)
        _, _, f1_cls = _pr_f1(tp, fp + mc, fn + mc)
        avg_cf = (sum(cf)/len(cf)) if cf else 0
        hay_gt = any(r.has_gt for r in rs)
        c_loc = f"{f1_loc:.3f}" if hay_gt else "—"
        c_cls = f"{f1_cls:.3f}" if hay_gt else "—"
        rows_models += (
            f"<tr><td>{slot.alias}</td><td class='r'>{n_img}</td><td class='r'>{n_det}</td>"
            f"<td class='r'>{avg_cf:.3f}</td>"
            f"<td class='r'>{tp}</td><td class='r'>{fp}</td><td class='r'>{fn}</td>"
            f"<td class='r'>{mc}</td>"
            f"<td class='r'>{c_loc}</td><td class='r'>{c_cls}</td></tr>"
        )

    # Nota que acompana la tabla, con las cifras del propio lote.
    nota_miscls = ""
    if any_gt:
        tp_g = sum(r.tp for r in all_results)
        fp_g = sum(r.fp for r in all_results)
        fn_g = sum(r.fn for r in all_results)
        p_loc, r_loc, f_loc = _pr_f1(tp_g, fp_g, fn_g)
        p_cls, r_cls, f_cls = _pr_f1(tp_g, fp_g + total_mc, fn_g + total_mc)
        nota_miscls = (
            "<p><strong>Los dos F1 miden cosas distintas.</strong> "
            f"<em>Localización</em> responde si el detector encuentra la partícula "
            f"(P {p_loc:.3f} · R {r_loc:.3f} · <strong>F1 {f_loc:.3f}</strong>). "
            f"<em>Con clase</em> exige además acertar el polímero, contando cada "
            f"caja mal clasificada como falso positivo de la clase predicha y "
            f"falso negativo de la real "
            f"(P {p_cls:.3f} · R {r_cls:.3f} · <strong>F1 {f_cls:.3f}</strong>).</p>"
            f"<p>La diferencia corresponde a <strong>{total_mc}</strong> "
            f"partícula(s) bien localizada(s) pero asignada(s) a la clase "
            f"incorrecta. Es la cifra que concilia esta tabla con la de "
            f"precisión por clase de la sección de errores.</p>")

    # ── Veredicto y barrido de confianza ──
    veredicto_html = ""
    if any_gt:
        barrido = _barrido_confianza(resultados, state, state.params.iou_tp,
                                     state.params.conf)
        if barrido:
            resumen = []
            for alias, filas in barrido.items():
                mejor = max(filas, key=lambda f: f[3])
                actual = filas[0]
                resumen.append((alias, actual, mejor))

            # Veredicto: gana quien tenga mejor F1 con clase en su mejor umbral.
            resumen.sort(key=lambda x: x[2][3], reverse=True)
            g_alias, g_act, g_mej = resumen[0]
            texto = (f"<p><strong>Mejor desempeño: {g_alias}</strong>, con "
                     f"F1 {g_mej[3]:.3f} al umbral {g_mej[0]:.2f} "
                     f"(P {g_mej[1]:.3f} · R {g_mej[2]:.3f}).</p>")
            if len(resumen) > 1:
                p_alias, _, p_mej = resumen[-1]
                d = g_mej[3] - p_mej[3]
                texto += (f"<p>La diferencia con {p_alias} es de "
                          f"<strong>{d:.3f}</strong> de F1. Con un solo "
                          f"entrenamiento por arquitectura, una diferencia "
                          f"pequeña no distingue el diseño de la red del azar "
                          f"de inicialización: haría falta repetir con distintas "
                          f"semillas para afirmar que una es superior.</p>")

            veredicto_html = texto

    # ── Comparación entre modelos, foto por foto ──
    # La tabla global de arriba dice cual modelo gana en total; esta dice en
    # cuales fotos gana. Con lotes desiguales como los del Loa, donde un tramo
    # concentra la mayoria de las particulas, el total lo decide ese tramo y
    # puede esconder que en el resto el otro modelo va mejor.
    compare_html = ""
    activos = [mi for mi, s in enumerate(state.model_slots) if s.path is not None]
    if len(activos) > 1:
        por_foto = {}
        for mi in activos:
            for r in resultados.get(mi, []):
                por_foto.setdefault(r.image_path, {})[mi] = r

        hay_gt = any(r.has_gt for rs in resultados.values() for r in rs)
        cab = "<tr><th>Imagen</th>"
        for mi in activos:
            alias = state.model_slots[mi].alias
            cab += (f"<th>{alias}<br><span style='font-weight:400;font-size:8.5pt'>dets</span></th>")
            if hay_gt:
                cab += (f"<th><span style='font-weight:400;font-size:8.5pt'>"
                        f"{LABEL_TP}/{LABEL_FP}/{LABEL_FN}</span></th>")
        cab += "</tr>"

        filas = ""
        for ruta in sorted(por_foto):
            filas += f"<tr><td>{Path(ruta).name}</td>"
            for mi in activos:
                r = por_foto[ruta].get(mi)
                if r is None:
                    filas += "<td class='r'>—</td>"
                    if hay_gt:
                        filas += "<td class='r'>—</td>"
                    continue
                filas += f"<td class='r'>{len(r.predictions)}</td>"
                if hay_gt:
                    celda = (f"{r.tp}/{r.fp}/{r.fn}" if r.has_gt else "—")
                    filas += f"<td class='r'>{celda}</td>"
            filas += "</tr>"

        # Veredicto: solo se declara si hay ground truth con que juzgar. Sin el,
        # mas detecciones no es mejor, y decir lo contrario seria enganoso.
        veredicto = ""
        if hay_gt:
            marcador = []
            for mi in activos:
                rs = resultados.get(mi, [])
                tp = sum(r.tp for r in rs); fp = sum(r.fp for r in rs)
                fn = sum(r.fn for r in rs)
                pr = tp/(tp+fp) if (tp+fp) else 0
                rc = tp/(tp+fn) if (tp+fn) else 0
                marcador.append((2*pr*rc/(pr+rc) if (pr+rc) else 0,
                                 state.model_slots[mi].alias))
            marcador.sort(reverse=True)
            mejor_f1, mejor = marcador[0]
            # Este F1 es el permisivo (localizacion). Decia solo "F1 global" y
            # convivia con el veredicto estricto de arriba: dos numeros distintos
            # llamados igual en la misma seccion.
            veredicto = (
                f"<p><b>Mejor F1 de localización:</b> {mejor} ({mejor_f1:.3f}) — "
                f"encontrar la partícula, sin exigir que acierte el polímero. El "
                f"veredicto con clase está más arriba.</p>")
        else:
            veredicto = ("<p class='caption' style='text-align:left'>Sin ground "
                         "truth no se puede declarar un ganador: un modelo con "
                         "más detecciones puede estar acertando o inventando. "
                         "Carga anotaciones para obtener F1 por modelo.</p>")

        # ── Figuras de comparación ──
        figuras_cmp = ""
        if hay_gt:
            datos_barras = []
            por_clase_modelo: Dict[str, Dict[str, tuple]] = {}
            for mi in activos:
                alias = state.model_slots[mi].alias
                rs = [r for r in resultados.get(mi, []) if r.has_gt]
                tp = sum(r.tp for r in rs); fp = sum(r.fp for r in rs)
                fn = sum(r.fn for r in rs); mc = sum(r.miscls for r in rs)
                p_l, r_l, f_l = _pr_f1(tp, fp, fn)
                p_c, r_c, f_c = _pr_f1(tp, fp + mc, fn + mc)
                datos_barras.append((alias, p_l, r_l, f_l, p_c, r_c, f_c))

                gts_m = [r.gt for r in rs]
                preds_m = [r.predictions for r in rs]
                ids = sorted({d.class_id for lst in gts_m + preds_m for d in lst})
                if ids:
                    metricas = aggregate_per_class(preds_m, gts_m, ids,
                                                   iou_thr=state.params.iou_tp)
                    nombres_cls = {
                        cid: next((d.class_name for lst in gts_m + preds_m
                                   for d in lst if d.class_id == cid), str(cid))
                        for cid in ids}
                    por_clase_modelo[alias] = {
                        nombres_cls[cid]: (metricas[cid].precision,
                                           metricas[cid].recall,
                                           metricas[cid].f1) for cid in ids}

            f_barras = _fig_comparacion_metricas(datos_barras)
            if f_barras:
                figuras_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_barras}' />"
                    f"<div class='caption'>Figura. Precisión, Recall y F1 de cada "
                    f"modelo al umbral {state.params.conf:g}, en los dos criterios. "
                    f"La distancia entre paneles es la confusión entre "
                    f"polímeros.</div></div>")

            f_clase = _fig_metricas_por_clase(por_clase_modelo)
            if f_clase:
                figuras_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_clase}' />"
                    f"<div class='caption'>Figura. F1 por clase. Un modelo puede "
                    f"ganar en el agregado y perder en el polímero que "
                    f"interesa.</div></div>")

            # Acuerdo foto a foto entre los dos primeros modelos.
            if len(activos) >= 2:
                a, b = activos[0], activos[1]
                pares = []
                for ruta in sorted(por_foto):
                    ra, rb = por_foto[ruta].get(a), por_foto[ruta].get(b)
                    if ra is not None and rb is not None:
                        pares.append((len(ra.predictions), len(rb.predictions),
                                      Path(ruta).name))
                f_acuerdo = _fig_acuerdo_por_imagen(
                    state.model_slots[a].alias, state.model_slots[b].alias, pares)
                if f_acuerdo:
                    figuras_cmp += (
                        f"<div class='fig'><img src='data:image/png;base64,{f_acuerdo}' />"
                        f"<div class='caption'>Figura. Detecciones por imagen de un "
                        f"modelo frente al otro; la diagonal es el acuerdo exacto. "
                        f"Escala simétrica-logarítmica, porque casi todas las "
                        f"placas tienen pocas partículas y unas pocas "
                        f"cientos.</div></div>")

            # Curva de F1 frente al umbral: sostiene la eleccion del punto de
            # operacion con evidencia, en vez de afirmarla.
            barrido = _barrido_confianza(resultados, state, state.params.iou_tp,
                                         state.params.conf)
            f_barrido = _fig_barrido(barrido)
            if f_barrido:
                figuras_cmp += (
                    f"<div class='fig'><img src='data:image/png;base64,{f_barrido}' />"
                    f"<div class='caption'>Figura. F1 (con clase) frente al umbral "
                    f"de confianza; la estrella marca el máximo de cada modelo. La "
                    f"curva arranca en {state.params.conf:g}, el umbral con que se "
                    f"ejecutó: por debajo las detecciones no se calcularon.</div></div>")

        compare_html = (
            "<p>Detecciones de cada modelo imagen por imagen. El total puede "
            "estar dominado por unas pocas fotos densas, así que conviene mirar "
            "el detalle antes de elegir modelo.</p>"
            + veredicto
            + figuras_cmp
            + f"<table class='data'>{cab}{filas}</table>")
    elif len(activos) == 1:
        compare_html = ("<p class='caption' style='text-align:left'>Solo se "
                        "ejecutó un modelo, así que no hay comparación. Carga un "
                        "segundo modelo en la pestaña Modelos para compararlos "
                        "sobre las mismas imágenes.</p>")

    # ── Galería comparativa: Predicción vs Ground Truth (lado a lado) ──
    gallery_html = ""
    omitidas = 0
    if include_gallery:
        blocks = []
        # Tope de imágenes en galería: cada una se incrusta en base64, así que
        # sin límite un lote de varios cientos produce un HTML de gigabytes.
        # Se agrupa por imagen y no por modelo. Aplanando modelo a modelo, el
        # tope se agotaba dentro del primero y el segundo no aparecia nunca en
        # la galeria: con dos modelos cargados solo se veian las fotos de uno,
        # que es justo la comparacion visual que se quiere. Ahora el tope cuenta
        # imagenes, y de cada una entran todos los modelos, uno bajo el otro.
        por_imagen: dict = {}
        for mi in sorted(resultados):
            for r in resultados[mi]:
                por_imagen.setdefault(r.image_path, []).append((mi, r))
        # Orden por estación/tramo/placa cuando el nombre de la placa lo
        # permite: el informe se lee siguiendo el testigo hacia abajo, no en el
        # orden arbitrario en que se cargaron los archivos.
        orden_imgs = sorted(por_imagen, key=_clave_orden)
        candidatas = []
        for ruta in orden_imgs[:max_gallery]:
            candidatas.extend(por_imagen[ruta])
        omitidas = max(0, len(por_imagen) - max_gallery)
        for mi, r in candidatas:
            slot = state.model_slots[mi]
            if True:
                # Imagen de predicción (preferimos pred_png; si no, la combinada)
                pred_bytes = getattr(r, "pred_png", None) or r.annotated_png
                if not pred_bytes:
                    continue
                left = (
                    f"<figure><img src='{_uri_galeria(pred_bytes)}' />"
                    f"<figcaption>Predicción del modelo · {slot.alias}"
                    f"<span class='sub'>{len(r.predictions)} detección(es) por YOLO</span>"
                    f"</figcaption></figure>"
                )

                # Imagen de Ground Truth (control)
                gt_bytes = getattr(r, "gt_png", None)
                if r.has_gt and gt_bytes:
                    right = (
                        f"<figure><img src='{_uri_galeria(gt_bytes)}' />"
                        f"<figcaption>Ground Truth (control)"
                        f"<span class='sub'>{len(r.gt)} etiqueta(s) reales</span>"
                        f"</figcaption></figure>"
                    )
                else:
                    right = (
                        "<figure><div class='nogt'>Sin Ground Truth para esta imagen</div>"
                        "<figcaption>Ground Truth (control)"
                        "<span class='sub'>no disponible</span></figcaption></figure>"
                    )

                # Encabezado de la foto: nombre y, si el nombre lo permite,
                # de qué tramo y placa viene.
                muestra = _muestra_de(r.image_path)
                donde = ""
                if muestra:
                    estacion, tramo, placa = muestra
                    donde = (f" &nbsp;<span class='tag'>{estacion} · tramo {tramo} "
                             f"· placa {placa}</span>")
                meta = (f"<strong>{r.image_path.name}</strong>{donde}"
                        f"&nbsp;<span class='tag'>modelo: {slot.alias}</span>")

                # Tabla de la foto: partículas por polímero. Los aciertos y
                # fallos emparejados caja a caja son otra pregunta y viven en
                # la sección de análisis de errores.
                tabla = _tabla_conteo(_conteo_por_clase(r.gt) if r.has_gt else None,
                                      _conteo_por_clase(r.predictions),
                                      clases_lote)

                blocks.append(
                    f"<div class='compare'><div class='compare-pair'>{left}{right}</div>"
                    f"<div class='compare-meta'>{meta}</div>{tabla}</div>"
                )

        if blocks:
            nota = ""
            if omitidas:
                n_mostradas = len({r.image_path for _, r in candidatas})
                nota = (f"<p class='caption'>Se muestran las primeras "
                        f"{n_mostradas} imágenes, cada una con todos los modelos; "
                        f"{omitidas} quedaron fuera de la galería para que el "
                        f"archivo siga siendo manejable. Las métricas de las "
                        f"secciones anteriores sí incluyen todas.</p>")
            gallery_html = (
                ""
                "<p>Cada bloque muestra, a la izquierda, las detecciones del modelo "
                "(<em>bounding boxes</em> dibujadas por YOLO con su clase y confianza) y, a la "
                "derecha, las etiquetas reales de control (<em>Ground Truth</em>). Esta vista "
                "lado a lado permite evaluar visualmente dónde acertó o falló el modelo.</p>"
                + nota + "".join(blocks)
            )

    # ── Figuras agregadas ──
    fig_classes = _fig_class_distribution(dict(per_class))
    fig_conf = _fig_confidence_hist(confs)
    fig_size = _fig_size_hist(sizes) if sizes else ""

    figures_html = ""
    if fig_classes:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_classes}' /><div class='caption'>Figura. Distribución de detecciones por clase.</div></div>"
    if fig_conf:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_conf}' /><div class='caption'>Figura. Histograma de confianza.</div></div>"
    if fig_size:
        figures_html += f"<div class='fig'><img src='data:image/png;base64,{fig_size}' /><div class='caption'>Figura. Distribución de tamaños (diámetro equivalente en μm).</div></div>"

    # ── Métodos ──
    p = state.params
    methods_rows = [
        ("Modelos cargados", ", ".join(s.alias for s in active) or "—"),
        ("Confianza mínima", f"{p.conf:g}"),
        ("IoU NMS", f"{p.iou_nms:g}"),
        ("IoU para emparejar Verdaderos Positivos", f"{p.iou_tp:g}"),
        ("Tamaño de imagen (imgsz)", f"{p.imgsz}"),
        ("Dispositivo", p.device),
        ("μm por píxel", f"{p.um_per_px:g}" if p.um_per_px > 0 else "—"),
        ("Filtro tamaño (μm)", f"{p.size_min_um} – {p.size_max_um}" if (p.size_min_um or p.size_max_um) else "sin filtro"),
        ("Imágenes procesadas", str(total_imgs)),
        ("Total de detecciones", str(total_dets)),
    ]
    methods_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in methods_rows)
    equipo_html = _seccion_equipo(active)

    # Párrafo de métodos redactado, listo para copiar al manuscrito
    # (reproducibilidad: modelo, versión y parámetros exactos).
    try:
        import ultralytics
        _ul_ver = ultralytics.__version__
    except Exception:
        _ul_ver = "—"
    _model_names = ", ".join(s.alias for s in active) if active else "—"
    methods_para = (
        "<p><strong>Resumen de la configuración empleada:</strong></p>"
        "<blockquote style='border-left:3px solid var(--accent); margin:8px 0; "
        "padding:6px 14px; color:#424a53; background:#f6f8fa; border-radius:0 6px 6px 0;'>"
        f"La detección automatizada se realizó con el modelo YOLO «{_model_names}» "
        f"(Ultralytics {_ul_ver}) a una resolución de entrada de {p.imgsz} px, "
        f"umbral de confianza {p.conf:g} y supresión de no-máximos con IoU {p.iou_nms:g}. "
        + (f"Las métricas de error se calcularon contra anotación manual independiente, "
           f"emparejando predicciones y etiquetas con IoU ≥ {p.iou_tp:g}. " if any_gt else "")
        + (f"La calibración óptica fue de {p.um_per_px:g} μm/píxel. " if p.um_per_px > 0 else "")
        + f"Se procesaron {total_imgs} imágenes con un total de "
        f"{total_dets} detecciones."
        "</blockquote>"
    )

    conteo_html = _seccion_conteo(resultados, state, active, clases_lote)

    refs_html = ""
    if include_refs:
        refs_html = """
        
        <ol>
          <li>Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).
              Development of a new methodology for the determination of PET microplastics in sediment,
              based on microwave-assisted acid digestion.
              <em>PLoS ONE</em> 19(12): e0314520.
              <a href='https://doi.org/10.1371/journal.pone.0314520'>https://doi.org/10.1371/journal.pone.0314520</a></li>
          <li>Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).
              Evaluation of microwave-assisted acid/oxidant digestion method for the detection of
              polyethylene microplastics in <em>Merluccius gayi</em> fish by Nile Red fluorescent
              staining and image analysis. <em>J. Chil. Chem. Soc.</em> 69(1): 6082-6085.
              <a href='https://doi.org/10.4067/s0717-97072024000106082'>https://doi.org/10.4067/s0717-97072024000106082</a></li>
        </ol>
        """

    # ── Calibración de escala ──
    # Se reporta la procedencia, no solo el número: un tamaño en µm sin decir
    # contra qué patrón se midió no es verificable. Si no hay calibración la
    # sección no se emite, en vez de mostrar un "—" que parece un dato.
    calib_html = ""
    cals = getattr(state, "calibraciones", None) or {}
    if cals:
        from .calibracion import resumen_lote, ORIGEN_PLACA
        res = resumen_lote(list(cals.values()))
        if res["n"]:
            etiqueta = {"placa": "medida sobre la placa Petri de esta foto",
                        "indice": "heredada del recorte (índice de calibración)",
                        "manual": "introducida a mano en Parámetros"}
            filas = "".join(
                f"<tr><td>{etiqueta.get(o, o)}</td><td>{k} imagen{'es' if k != 1 else ''}</td></tr>"
                for o, k in sorted(res["origenes"].items()))
            aviso = ""
            if res["variacion"] > 1.05:
                aviso = (f"<p><strong>La escala no es común a todo el lote:</strong> varía "
                         f"{res['variacion']:.2f}× entre la foto más cercana y la más lejana. "
                         f"Por eso cada imagen se convierte con su propio factor; usar uno "
                         f"solo para todas daría tamaños con hasta un "
                         f"{100 * (res['variacion'] - 1):.0f}% de error.</p>")
            calib_html = f"""
<table class='data'><tr><th>Procedencia de la escala</th><th>Imágenes</th></tr>{filas}</table>
<table class='data'>
<tr><th>µm por píxel</th><th>Valor</th></tr>
<tr><td>mínimo</td><td>{res['min']:.4f}</td></tr>
<tr><td>mediana</td><td>{res['mediana']:.4f}</td></tr>
<tr><td>máximo</td><td>{res['max']:.4f}</td></tr>
</table>
{aviso}
<p>El patrón de longitud es el diámetro externo nominal de la placa Petri. El radio
en píxeles se obtiene ajustando un círculo por mínimos cuadrados al borde del anillo
muestreado en 720 direcciones, con rechazo de atípicos; la transformada de Hough solo
aporta el centro aproximado, porque su radio llega a errar un 12&nbsp;% y ese error
entraría entero en todos los tamaños reportados.</p>"""
            if res["avisos"]:
                items = "".join(f"<li>{a}</li>" for a in res["avisos"][:12] if a)
                if items:
                    calib_html += f"<p>Incidencias durante la calibración:</p><ul>{items}</ul>"
    elif p.um_per_px > 0:
        calib_html = (
            f"<p>Escala única para todo el lote, introducida a mano: "
            f"<strong>{p.um_per_px:g} µm/píxel</strong>. No se midió ninguna placa, "
            f"de modo que este factor no está trazado a un patrón de longitud y "
            f"cualquier variación en la distancia de disparo entre fotos queda sin "
            f"corregir.</p>")

    # ── Forma y talla ──
    # Se separa de "Resultados" porque responde otra pregunta: no cuantas
    # particulas hay, sino que forma tienen. El morfotipo y la curvatura son
    # variables que la literatura de microplasticos reporta y que la caja del
    # detector no puede dar.
    forma_html = ""
    formas = [p_ for r in all_results for p_ in r.predictions
              if getattr(p_, "aspecto", None)]
    # Las que cayeron al tamano de la caja: su talla no es comparable y hay que
    # decirlo, no promediarla en silencio con las medidas sobre mascara.
    n_sin_forma = sum(1 for r in all_results for p_ in r.predictions
                      if getattr(p_, "aspecto", None) is None)
    if formas:
        largos = [f.largo_um for f in formas if f.largo_um]
        curvas = [f.curvatura for f in formas if f.curvatura]
        fibras = sum(1 for f in formas if f.morfotipo == "fibra")
        n_curvas = sum(1 for c in curvas if c >= 1.15)
        filas_t = ""
        if largos:
            a = np.array(largos, dtype=float)
            filas_t = (f"<tr><td>Largo (µm)</td><td>{np.percentile(a,10):.0f}</td>"
                       f"<td>{np.median(a):.0f}</td><td>{np.percentile(a,90):.0f}</td></tr>")
            anchos = [f.ancho_um for f in formas if f.ancho_um]
            if anchos:
                b = np.array(anchos, dtype=float)
                filas_t += (f"<tr><td>Ancho (µm)</td><td>{np.percentile(b,10):.0f}</td>"
                            f"<td>{np.median(b):.0f}</td><td>{np.percentile(b,90):.0f}</td></tr>")
        pc_f = 100.0 * fibras / len(formas)
        pc_c = 100.0 * n_curvas / len(curvas) if curvas else 0.0
        forma_html = f"""
<table class='data'><tr><th>Morfotipo</th><th>Partículas</th><th>%</th></tr>
<tr><td>Fibra (relación de aspecto ≥ 3)</td><td>{fibras}</td><td>{pc_f:.1f} %</td></tr>
<tr><td>Fragmento</td><td>{len(formas) - fibras}</td><td>{100 - pc_f:.1f} %</td></tr>
</table>
<table class='data'><tr><th>Dimensión</th><th>p10</th><th>mediana</th><th>p90</th></tr>{filas_t}</table>
<p>Las magnitudes se miden sobre la <strong>máscara de cada partícula</strong>, no sobre su
caja. La caja de una partícula alargada está casi vacía y depende de cómo haya caído: una
fibra tumbada en diagonal tiene caja cuadrada, de modo que medir sobre la caja la reportaría
como fragmento y con una talla equivocada.</p>
<p>El largo se obtiene resolviendo el rectángulo de igual área y perímetro,
<em>L</em> = (<em>P</em> + √(<em>P</em>²−16<em>A</em>))&nbsp;/&nbsp;4. Doblar una fibra no
cambia ni su área ni su perímetro, así que este largo sigue la curva sin necesidad de
modelarla. En partículas compactas el discriminante es negativo —no existe tal rectángulo— y
entonces se informa la extensión recta.</p>
<p><strong>{n_curvas} partículas ({pc_c:.1f} %) están curvadas</strong>, entendiendo por tal que
su largo supera en más de un 15 % su extensión en línea recta. En ellas la distancia entre
extremos subestima la talla, y es la razón por la que no se usa.</p>"""
        if n_sin_forma:
            forma_html += (
                f"<p>En {n_sin_forma} partículas no se pudo separar la partícula del fondo; "
                f"su talla proviene de la caja y no es comparable con el resto.</p>")

    # ── Ensamblar HTML ──
    # Cada bloque es (id, cuerpo). El cuerpo NO lleva su <h2>: el titulo y el
    # numero los pone el ensamblador, que es el unico que sabe que secciones
    # sobrevivieron al filtro y por tanto que numero le toca a cada una.
    cuerpos = {
        "resumen": f"""
<div class='kpis'>
  <div class='kpi'><div class='l'>Imágenes</div><div class='v'>{total_imgs}</div><div class='b' style='background:var(--accent)'></div></div>
  <div class='kpi'><div class='l'>Detecciones</div><div class='v'>{total_dets}</div><div class='b' style='background:var(--accent)'></div></div>
  <div class='kpi'><div class='l'>Conf. media</div><div class='v'>{avg_conf:.3f}</div><div class='b' style='background:var(--vio)'></div></div>
  <div class='kpi'><div class='l'>Tamaño medio (μm)</div><div class='v'>{(f"{avg_size:.1f}" if avg_size else "—")}</div><div class='b' style='background:var(--warn)'></div></div>
</div>
<p>Se analizaron <strong>{total_imgs} imágenes</strong> con
<strong>{len(active)} modelo{'s' if len(active)!=1 else ''}</strong> YOLO entrenado para detectar
microplásticos de PET, PP y LDPE bajo fluorescencia Nile Red (254 nm). El total de detecciones
fue <strong>{total_dets}</strong> con una confianza media de <strong>{avg_conf:.3f}</strong>.
{"Se incluyó análisis de errores con Ground Truth (Verdaderos Positivos, Falsos Positivos, Falsos Negativos y Mal Clasificados)." if any_gt else "No se aportó Ground Truth, por lo que no se reportan métricas de error."}
</p>""",
        "metodos": f"""
<table class='data'><tr><th>Parámetro</th><th>Valor</th></tr>{methods_html}</table>
{equipo_html}
{methods_para}""",
        "calibracion": calib_html,
        "forma": forma_html,
        "resultados": figures_html,
        "modelos": f"""
<table class='data'><tr><th>Modelo</th><th>Imágenes</th><th>Detecciones</th>
<th>Conf. media</th><th>{LABEL_TP}</th><th>{LABEL_FP}</th><th>{LABEL_FN}</th>
<th>{LABEL_MISCLS}</th><th>F1<br><span style='font-weight:400;font-size:8.5pt'>localización</span></th>
<th>F1<br><span style='font-weight:400;font-size:8.5pt'>con clase</span></th></tr>{rows_models}</table>
{nota_miscls}""",
        "errores": err_section,
        "comparacion": f"{veredicto_html}\n{compare_html}",
        "galeria": gallery_html,
        "conteo": conteo_html,
        "referencias": refs_html,
    }

    # Un id en ``pedidas`` con cuerpo vacio se cae aqui: se marco la casilla pero
    # no habia con que llenarla (errores sin ground truth, conteo sin muestras).
    ancla = {"resumen": "abstract", "metodos": "methods", "calibracion": "calib",
             "forma": "forma",
             "resultados": "results", "modelos": "models", "errores": "errors",
             "comparacion": "compare", "galeria": "gallery", "conteo": "conteo",
             "referencias": "refs"}
    vivas = [(sid, titulo, cuerpos.get(sid, ""))
             for sid, titulo in SECCIONES
             if sid in pedidas and (cuerpos.get(sid) or "").strip()]

    toc_html = "".join(
        f"<li><a href='#{ancla[sid]}'>{titulo}</a></li>" for sid, titulo, _ in vivas)
    secciones_html = "".join(
        f"<h2 id='{ancla[sid]}'>{i}. {titulo}</h2>\n{cuerpo.replace(NUM, str(i))}\n"
        for i, (sid, titulo, cuerpo) in enumerate(vivas, start=1))

    html = f"""<!doctype html>
<html lang='es'><head><meta charset='utf-8'>
<title>Informe de detección · Poly-X</title>
<style>{REPORT_CSS}</style></head><body>
<div class='container'>

<header class='cover'>
  <div class='kicker'>Poly-X · Informe de detección</div>
  <h1>Informe de detección de microplásticos<br>por fluorescencia Nile Red</h1>
  <p class='meta'><strong>Autor:</strong> Cristofher Ferrada &middot;
    <strong>Modelos:</strong> {', '.join(s.alias for s in active) or '—'}</p>
</header>

<div class='toc'>
  <h3>📑 Índice</h3>
  <ol>{toc_html}</ol>
</div>

{secciones_html}
<footer>
  © Cristofher Ferrada · Generado por Poly-X<br>
  Suite de detección de microplásticos por fluorescencia Nile Red (254 nm) e IA (YOLO v8/v11).
</footer>
</div></body></html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
