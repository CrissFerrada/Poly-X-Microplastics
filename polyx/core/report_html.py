"""Generador de reporte HTML paper-quality (autocontenido, imágenes en base64).

Contenido:
  1. Abstract (resumen ejecutivo)
  2. Métodos (modelo, params, calibración, dispositivo, fecha)
  3. Resultados generales (clase, conf, tamaño, P/R/F1)
  4. Resumen por modelo
  5. Análisis de errores (matriz de confusión + galería)
  6. Comparación entre modelos
  7. Galería completa por imagen anotada
  8. Referencias bibliográficas
"""
from __future__ import annotations
import base64
import io
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
h2{page-break-after:avoid;}.fig,.gallery .item{page-break-inside:avoid;}}
"""


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


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


# ────────────────────────────────────────────────────────────────────
def generate_report(state, output_path: Path,
                    include_refs: bool = True,
                    include_gallery: bool = True,
                    max_gallery: int = 60,
                    solo_imagenes=None) -> Path:
    """Genera el reporte HTML. `state` es un DetectorState con resultados.

    ``max_gallery`` limita cuántas imágenes se incrustan en la galería. Las
    imágenes van en base64 dentro del propio HTML, de modo que sin tope un lote
    grande generaba un archivo inabrible (14.5 MB con solo 4 imágenes).

    ``solo_imagenes`` acota el informe a un conjunto de rutas. Con ``None`` entra
    el trabajo completo. El filtro se aplica antes de calcular nada, de forma que
    los totales, los gráficos y la matriz de confusión describan exactamente el
    mismo subconjunto que se muestra; si se filtrara solo la galería, las cifras
    de arriba hablarían de un lote y las imágenes de otro.
    """
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
        err_section = f"""
        <h2 id='errors'>5. Análisis de errores</h2>
        <h3>5.1 Matriz de confusión</h3>
        <div class='fig'><img src='data:image/png;base64,{cm_img}' />
            <div class='caption'>Figura. Matriz de confusión (modelo principal: {active[0].alias}, IoU = {state.params.iou_tp}).</div></div>
        <h3>5.2 Precisión / Recall / F1 por clase</h3>
        <table class='data'><tr><th>Clase</th><th>{LABEL_TP}</th><th>{LABEL_FP}</th><th>{LABEL_FN}</th>
        <th>Precisión</th><th>Recall</th><th>F1</th></tr>{rows}</table>
        """

    # ── Resumen por modelo (tabla comparativa) ──
    rows_models = ""
    for mi, slot in enumerate(state.model_slots):
        if slot.path is None: continue
        rs = resultados.get(mi, [])
        n_img = len({r.image_path for r in rs})
        n_det = sum(len(r.predictions) for r in rs)
        cf = [p.conf for r in rs for p in r.predictions]
        tp = sum(r.tp for r in rs); fp = sum(r.fp for r in rs); fn = sum(r.fn for r in rs)
        prec = tp/(tp+fp) if (tp+fp) else 0; rec = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        avg_cf = (sum(cf)/len(cf)) if cf else 0
        f1_cell = f"{f1:.3f}" if any(r.has_gt for r in rs) else "—"
        rows_models += (
            f"<tr><td>{slot.alias}</td><td class='r'>{n_img}</td><td class='r'>{n_det}</td>"
            f"<td class='r'>{avg_cf:.3f}</td>"
            f"<td class='r'>{tp}</td><td class='r'>{fp}</td><td class='r'>{fn}</td>"
            f"<td class='r'>{f1_cell}</td></tr>"
        )

    # ── Galería comparativa: Predicción vs Ground Truth (lado a lado) ──
    gallery_html = ""
    omitidas = 0
    if include_gallery:
        blocks = []
        # Tope de imágenes en galería: cada una se incrusta en base64, así que
        # sin límite un lote de varios cientos produce un HTML de gigabytes.
        candidatas = [(mi, r) for mi, rs in resultados.items() for r in rs]
        omitidas = max(0, len(candidatas) - max_gallery)
        for mi, r in candidatas[:max_gallery]:
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

                # Métricas por imagen (con nombres completos en español)
                if r.has_gt:
                    meta = (
                        f"<strong>{r.image_path.name}</strong> &nbsp; "
                        f"<span class='tag'>{LABEL_TP}: {r.tp}</span>"
                        f"<span class='tag'>{LABEL_FP}: {r.fp}</span>"
                        f"<span class='tag'>{LABEL_FN}: {r.fn}</span>"
                        f"<span class='tag'>{LABEL_MISCLS}: {r.miscls}</span>"
                    )
                else:
                    meta = (
                        f"<strong>{r.image_path.name}</strong> &nbsp; "
                        f"<span class='tag'>{len(r.predictions)} detección(es)</span>"
                        f"<span class='tag'>sin Ground Truth</span>"
                    )

                blocks.append(
                    f"<div class='compare'><div class='compare-pair'>{left}{right}</div>"
                    f"<div class='compare-meta'>{meta}</div></div>"
                )

        if blocks:
            nota = ""
            if omitidas:
                nota = (f"<p class='caption'>Se muestran las primeras {max_gallery} "
                        f"imágenes; {omitidas} quedaron fuera de la galería para que "
                        f"el archivo siga siendo manejable. Las métricas de las "
                        f"secciones anteriores sí incluyen todas.</p>")
            gallery_html = (
                "<h2 id='gallery'>7. Galería comparativa: Predicción vs Ground Truth</h2>"
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
        ("Confianza mínima", f"{p.conf}"),
        ("IoU NMS", f"{p.iou_nms}"),
        ("IoU para emparejar Verdaderos Positivos", f"{p.iou_tp}"),
        ("Tamaño de imagen (imgsz)", f"{p.imgsz}"),
        ("Dispositivo", p.device),
        ("μm por píxel", f"{p.um_per_px}" if p.um_per_px > 0 else "—"),
        ("Filtro tamaño (μm)", f"{p.size_min_um} – {p.size_max_um}" if (p.size_min_um or p.size_max_um) else "sin filtro"),
        ("Imágenes procesadas", str(total_imgs)),
        ("Total de detecciones", str(total_dets)),
        ("Fecha de generación", now),
    ]
    methods_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in methods_rows)

    # Párrafo de métodos redactado, listo para copiar al manuscrito
    # (reproducibilidad: modelo, versión, parámetros y fecha exactos).
    try:
        import ultralytics
        _ul_ver = ultralytics.__version__
    except Exception:
        _ul_ver = "—"
    _model_names = ", ".join(s.alias for s in active) if active else "—"
    methods_para = (
        "<p><strong>Texto sugerido para la sección de métodos del manuscrito:</strong></p>"
        "<blockquote style='border-left:3px solid var(--accent); margin:8px 0; "
        "padding:6px 14px; color:#424a53; background:#f6f8fa; border-radius:0 6px 6px 0;'>"
        f"La detección automatizada se realizó con el modelo YOLO «{_model_names}» "
        f"(Ultralytics {_ul_ver}) a una resolución de entrada de {p.imgsz} px, "
        f"umbral de confianza {p.conf} y supresión de no-máximos con IoU {p.iou_nms}. "
        + (f"Las métricas de error se calcularon contra anotación manual independiente, "
           f"emparejando predicciones y etiquetas con IoU ≥ {p.iou_tp}. " if any_gt else "")
        + (f"La calibración óptica fue de {p.um_per_px} μm/píxel. " if p.um_per_px > 0 else "")
        + f"Se procesaron {total_imgs} imágenes con un total de {total_dets} detecciones "
        f"(análisis del {now})."
        "</blockquote>"
    )

    refs_html = ""
    if include_refs:
        refs_html = """
        <h2 id='refs'>8. Referencias bibliográficas</h2>
        <ol>
          <li>Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).
              Development of a new methodology for the determination of PET microplastics in sediment,
              based on microwave-assisted acid digestion.
              <em>PLoS ONE</em> 19(12): e0314520.
              <a href='https://doi.org/10.1371/journal.pone.0314520'>https://doi.org/10.1371/journal.pone.0314520</a></li>
          <li>Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).
              Evaluation of microwave-assisted acid/oxidant digestion method for the detection of
              polyethylene microplastics in <em>Merluccius gayi</em> fish by Nile Red fluorescent
              staining and image analysis. <em>J. Chil. Chem. Soc.</em> 69(1): 6082.</li>
        </ol>
        """

    # ── Ensamblar HTML ──
    html = f"""<!doctype html>
<html lang='es'><head><meta charset='utf-8'>
<title>Reporte Poly-X · {now}</title>
<style>{REPORT_CSS}</style></head><body>
<div class='container'>

<header class='cover'>
  <div class='kicker'>Poly-X · Reporte paper-quality</div>
  <h1>Análisis automatizado de microplásticos<br>por fluorescencia Nile Red e IA</h1>
  <p class='meta'><strong>Autor:</strong> Cristofher Ferrada &middot;
    <strong>Generado:</strong> {now} &middot;
    <strong>Modelos:</strong> {', '.join(s.alias for s in active) or '—'}</p>
</header>

<div class='toc'>
  <h3>📑 Índice</h3>
  <ol>
    <li><a href='#abstract'>Resumen</a></li>
    <li><a href='#methods'>Métodos</a></li>
    <li><a href='#results'>Resultados generales</a></li>
    <li><a href='#models'>Resumen por modelo</a></li>
    {"<li><a href='#errors'>Análisis de errores</a></li>" if any_gt else ""}
    <li><a href='#compare'>Comparación entre modelos</a></li>
    {"<li><a href='#gallery'>Galería por imagen</a></li>" if gallery_html else ""}
    {"<li><a href='#refs'>Referencias</a></li>" if include_refs else ""}
  </ol>
</div>

<h2 id='abstract'>1. Resumen</h2>
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
</p>

<h2 id='methods'>2. Métodos</h2>
<table class='data'><tr><th>Parámetro</th><th>Valor</th></tr>{methods_html}</table>
{methods_para}

<h2 id='results'>3. Resultados generales</h2>
{figures_html}

<h2 id='models'>4. Resumen por modelo</h2>
<table class='data'><tr><th>Modelo</th><th>Imágenes</th><th>Detecciones</th>
<th>Conf. media</th><th>{LABEL_TP}</th><th>{LABEL_FP}</th><th>{LABEL_FN}</th><th>F1</th></tr>{rows_models}</table>

{err_section}

<h2 id='compare'>6. Comparación entre modelos</h2>
<p>Esta sección consolida las métricas globales de cada modelo (tabla anterior) y permite
identificar cuál ofrece el mejor balance de precisión y cobertura sobre este conjunto de imágenes.</p>

{gallery_html}

{refs_html}

<footer>
  © Cristofher Ferrada · Generado por Poly-X · {now}<br>
  Suite de detección de microplásticos por fluorescencia Nile Red (254 nm) e IA (YOLO v8/v11).
</footer>
</div></body></html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
