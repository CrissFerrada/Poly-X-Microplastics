"""Hilo de inferencia en background. Procesa todas las imágenes con todos los modelos."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import io

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
    out = img_bgr.copy()
    for d in dets:
        c = color_for(d)
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = label_for(d)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), c, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


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

            for img_path in state.images:
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
                    has_gt = bool(gts)
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
                        """Codifica a PNG, lo guarda en disco y devuelve los bytes."""
                        if image is None:
                            return None
                        is_ok, buf = cv2.imencode(".png", image)
                        if not is_ok:
                            return None
                        data = bytes(buf)
                        (sub / f"{img_path.stem}{suffix}.png").write_bytes(data)
                        return data

                    annotated_bytes = _encode_and_save(combined, "_annot")
                    pred_bytes = _encode_and_save(pred_img, "_pred")
                    gt_bytes = _encode_and_save(gt_img, "_gt") if has_gt else None

                    res = ImageResult(
                        image_path=img_path, model_idx=real_idx,
                        predictions=preds, gt=gts, has_gt=has_gt,
                        tp=tp, fp=fp, fn=fn, miscls=mc,
                        annotated_png=annotated_bytes,
                        pred_png=pred_bytes,
                        gt_png=gt_bytes,
                        troceo=troceo_desc,
                    )
                    state.results.setdefault(real_idx, []).append(res)
                    self.image_done.emit(real_idx, res)

                    done += 1
                    self.progress.emit(done, total, img_path.name)

            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
