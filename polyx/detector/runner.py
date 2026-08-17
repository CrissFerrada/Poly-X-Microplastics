"""Hilo de inferencia en background. Procesa todas las imágenes con todos los modelos."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import io
import json

import numpy as np
import cv2
from PySide6.QtCore import QThread, Signal

from ..core.yolo_wrap import (
    YoloModel, Detection, find_gt_for_image, read_yolo_txt, compute_box_size_um,
)
from ..core.metrics import match_image
from ..core import theme as T
from .state import DetectorState, ImageResult


def _draw_annotated(img_bgr: np.ndarray, dets, color_for, label_for) -> np.ndarray:
    """Dibuja las cajas con un grosor proporcional al tamano de la imagen.

    Con grosor fijo de 2 px las anotaciones se volvian invisibles en las placas
    completas: una foto de 4096 px se reescala a 1100 para incrustarla en el
    informe, asi que esa linea acababa midiendo medio pixel. Escalando con el
    lado mayor, una placa se dibuja con ~6 px (lo mismo que usa
    paper/recomponer_placas.py) y un recorte de 1630 px conserva sus 2.
    """
    out = img_bgr.copy()
    lado = max(out.shape[:2])
    grosor = max(2, int(round(lado / 700)))
    escala_txt = max(0.5, lado / 3000)
    grosor_txt = max(1, int(round(escala_txt * 2)))

    for d in dets:
        c = color_for(d)
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, grosor)
        label = label_for(d)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          escala_txt, grosor_txt)
            margen = max(4, grosor * 2)
            cv2.rectangle(out, (x1, y1 - th - margen), (x1 + tw + margen, y1), c, -1)
            cv2.putText(out, label, (x1 + margen // 2, y1 - margen // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, escala_txt, (255, 255, 255),
                        grosor_txt, cv2.LINE_AA)
    return out


def _guardar_labels(dets, ruta_txt: Path, W: int, H: int) -> None:
    """Escribe las predicciones en formato YOLO dentro de la carpeta del run.

    Nunca junto a la foto: ahi vive el .txt del conteo manual, y volcar las
    predicciones encima destruiria el ground truth de forma irrecuperable.

    La confianza va como sexta columna. ``read_yolo_txt`` ignora las columnas de
    mas, asi que el archivo se puede reabrir en el Etiquetador como
    pre-anotacion -- contar a mano 500 particulas partiendo de las cajas del
    modelo es otra tarea que hacerlo desde cero -- y a la vez queda el registro
    de con que confianza salio cada caja.
    """
    lineas = []
    for d in dets:
        cx = ((d.x1 + d.x2) / 2) / W
        cy = ((d.y1 + d.y2) / 2) / H
        w = (d.x2 - d.x1) / W
        h = (d.y2 - d.y1) / H
        lineas.append(f"{d.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {d.conf:.4f}")
    ruta_txt.parent.mkdir(parents=True, exist_ok=True)
    ruta_txt.write_text("\n".join(lineas), encoding="utf-8")


def _color_bgr_for_class(name: str):
    hex_ = T.CLASS_COLOR_HEX.get(name, "#33aaff").lstrip("#")
    r, g, b = int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)
    return (b, g, r)


class DetectorRunner(QThread):
    progress = Signal(int, int, str)
    image_done = Signal(int, object)     # model_idx, ImageResult
    finished_ok = Signal()
    aborted = Signal()
    failed = Signal(str)

    def __init__(self, state: DetectorState, parent=None):
        super().__init__(parent)
        self.state = state

    def run(self):
        try:
            state = self.state
            params = state.params
            slots = state.active_models()
            if not slots:
                self.failed.emit("No hay modelos cargados.")
                return
            if not state.images:
                self.failed.emit("No hay imágenes seleccionadas.")
                return

            # Crear carpeta de run
            run_dir = Path(__file__).resolve().parents[2] / "runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state.run_dir = run_dir / f"detect_{stamp}"
            state.run_dir.mkdir(parents=True, exist_ok=True)

            # Cargar modelos
            for s in slots:
                if s.loaded is None:
                    s.loaded = YoloModel(str(s.path), alias=s.alias)
                s.loaded.load()

            # Resetear resultados
            state.results = {i: [] for i in range(len(state.model_slots))}
            total = len(state.images) * len(slots)
            done = 0

            for idx_img, img_path in enumerate(state.images):
                if state.consume_abort():
                    self.aborted.emit()
                    return

                # Cargar imagen una vez para GT/anotación
                img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    done += len(slots)
                    self.progress.emit(done, total, img_path.name)
                    continue
                H, W = img_bgr.shape[:2]

                # Cargar GT si existe (usa nombres del primer modelo si está disponible)
                names_first = slots[0].loaded.names if slots[0].loaded else {}
                gt_txt = find_gt_for_image(img_path, state.gt_folder)
                gts: list[Detection] = []
                if gt_txt is not None:
                    gts = read_yolo_txt(gt_txt, W, H, names_first)
                    # El tamaño del GT no depende del modelo → se calcula una vez
                    for d in gts:
                        compute_box_size_um(d, params.um_per_px if params.um_per_px > 0 else None)

                for mi_idx, slot in enumerate(slots):
                    # mi_idx es el orden dentro de active_models()
                    # necesitamos el índice real del slot dentro de model_slots
                    real_idx = state.model_slots.index(slot)
                    plan_troceo: dict = {}
                    preds = slot.loaded.predict_auto(
                        str(img_path),
                        conf=params.conf,
                        iou=params.iou_nms,
                        imgsz=params.imgsz,
                        device=params.device,
                        troceo=params.troceo,
                        umbral_px=params.troceo_umbral_px,
                        tile=params.troceo_tile,
                        overlap=params.troceo_overlap,
                        registro=plan_troceo,
                    )
                    # Se guarda la decision, no se registra de paso: el informe
                    # tiene que poder declarar si esa foto se conto troceada.
                    if plan_troceo.get("troceado"):
                        p = plan_troceo["plan"]
                        troceo_desc = (
                            f"{plan_troceo['ancho']}×{plan_troceo['alto']} px → "
                            f"{p['n_tiles']} tiles de {p['tile']} px, "
                            f"{int(p['overlap'] * 100)}% solape"
                        )
                    else:
                        troceo_desc = ""

                    # Calcular tamaño de las predicciones (el GT ya se calculó arriba)
                    for d in preds:
                        compute_box_size_um(d, params.um_per_px if params.um_per_px > 0 else None)

                    # Filtro por tamaño
                    if params.um_per_px > 0 and (params.size_min_um > 0 or params.size_max_um > 0):
                        keep = []
                        for d in preds:
                            if d.diam_um is None:
                                keep.append(d); continue
                            if params.size_min_um > 0 and d.diam_um < params.size_min_um: continue
                            if params.size_max_um > 0 and d.diam_um > params.size_max_um: continue
                            keep.append(d)
                        preds = keep

                    # Match con GT
                    # El .txt existe = hay ground truth, aunque no tenga
                    # ninguna caja: una placa revisada con cero particulas es un
                    # dato, no una placa sin revisar. Con bool(gts) esas placas
                    # quedaban fuera de las metricas y las detecciones que el
                    # modelo hacia sobre ellas no contaban como falso positivo.
                    has_gt = gt_txt is not None
                    if has_gt:
                        m = match_image(preds, gts, iou_thr=params.iou_tp)
                        tp, fp, fn, mc = m.tp, m.fp, m.fn, m.miscls
                    else:
                        tp = fp = fn = mc = 0

                    # ── Imágenes anotadas ──────────────────────────────
                    # 1) Predicciones del modelo (cajas de YOLO en color de clase).
                    pred_img = _draw_annotated(
                        img_bgr, preds,
                        color_for=lambda d: _color_bgr_for_class(d.class_name),
                        label_for=lambda d: f"{d.class_name} {d.conf:.2f}",
                    )
                    # 2) Combinada (predicción + GT en cian) para preview e
                    #    inspección de errores. Si no hay GT, coincide con pred.
                    combined = pred_img
                    gt_img = None
                    if has_gt:
                        combined = _draw_annotated(
                            pred_img, gts,
                            color_for=lambda d: (255, 200, 80),
                            label_for=lambda d: f"GT {d.class_name}",
                        )
                        # 3) Ground Truth solo (cajas reales en color de clase)
                        #    sobre la imagen original → comparación lado a lado.
                        gt_img = _draw_annotated(
                            img_bgr, gts,
                            color_for=lambda d: _color_bgr_for_class(d.class_name),
                            label_for=lambda d: f"{d.class_name}",
                        )

                    # ── Guardar PNGs en run_dir ────────────────────────
                    sub = state.run_dir / slot.alias
                    sub.mkdir(parents=True, exist_ok=True)

                    def _encode_and_save(image, suffix: str):
                        """Codifica a PNG, lo guarda en disco y devuelve su ruta.

                        Devuelve la ruta y no los bytes a proposito: en un lote
                        grande retener cada PNG en memoria llegaba a varios GB, y
                        el archivo ya queda escrito aqui mismo.
                        """
                        if image is None:
                            return None
                        is_ok, buf = cv2.imencode(".png", image)
                        if not is_ok:
                            return None
                        # El nombre lleva el indice de la imagen porque dos fotos
                        # de carpetas distintas pueden compartir nombre (1.1.jpg
                        # existe en varios testigos) y se pisarian entre si.
                        destino = sub / f"{idx_img:04d}_{img_path.stem}{suffix}.png"
                        destino.write_bytes(bytes(buf))
                        return destino

                    annotated_path = _encode_and_save(combined, "_annot")
                    pred_path = _encode_and_save(pred_img, "_pred")
                    gt_path = _encode_and_save(gt_img, "_gt") if has_gt else None

                    # Las cajas en texto, no solo dibujadas: un PNG no se puede
                    # reanalizar ni reabrir para corregir.
                    _guardar_labels(
                        preds,
                        sub / "labels" / f"{idx_img:04d}_{img_path.stem}.txt",
                        W, H)

                    res = ImageResult(
                        image_path=img_path, model_idx=real_idx,
                        predictions=preds, gt=gts, has_gt=has_gt,
                        tp=tp, fp=fp, fn=fn, miscls=mc,
                        annotated_path=annotated_path,
                        pred_path=pred_path,
                        gt_path=gt_path,
                        troceo=troceo_desc,
                    )
                    state.results.setdefault(real_idx, []).append(res)
                    self.image_done.emit(real_idx, res)

                    done += 1
                    self.progress.emit(done, total, img_path.name)

            self._escribir_resumen(state, slots)
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _escribir_resumen(self, state, slots) -> None:
        """Deja el run autodescrito: parametros y totales por modelo.

        Sin esto una carpeta de run son PNGs sueltos y no se puede saber con que
        confianza ni con que peso salieron. No se falla la corrida si el volcado
        no se puede escribir: los resultados ya estan en memoria y en disco.
        """
        p = state.params
        datos = {
            "parametros": {
                "confianza": p.conf, "iou_nms": p.iou_nms, "iou_tp": p.iou_tp,
                "imgsz": p.imgsz, "device": p.device, "um_por_px": p.um_per_px,
                "troceo": p.troceo, "troceo_umbral_px": p.troceo_umbral_px,
            },
            "imagenes": len(state.images),
            "modelos": [],
        }
        for slot in slots:
            mi = state.model_slots.index(slot)
            rs = state.results.get(mi, [])
            con_gt = [r for r in rs if r.has_gt]
            tp = sum(r.tp for r in con_gt); fp = sum(r.fp for r in con_gt)
            fn = sum(r.fn for r in con_gt); mc = sum(r.miscls for r in con_gt)
            datos["modelos"].append({
                "alias": slot.alias,
                "peso": str(slot.path) if slot.path else "",
                "detecciones": sum(len(r.predictions) for r in rs),
                "imagenes_con_gt": len(con_gt),
                "tp": tp, "fp": fp, "fn": fn, "mal_clasificados": mc,
            })
            # classes.txt junto a los labels, para poder reabrirlos como
            # pre-anotacion en el Etiquetador sin adivinar el orden de clases.
            nombres = getattr(slot.loaded, "names", None) or {}
            if nombres:
                destino = state.run_dir / slot.alias / "classes.txt"
                try:
                    destino.parent.mkdir(parents=True, exist_ok=True)
                    destino.write_text(
                        "\n".join(nombres[k] for k in sorted(nombres)),
                        encoding="utf-8")
                except OSError:
                    pass
        try:
            (state.run_dir / "resumen.json").write_text(
                json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
