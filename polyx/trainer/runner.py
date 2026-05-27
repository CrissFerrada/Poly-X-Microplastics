"""Hilo de entrenamiento en background con Ultralytics YOLO."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QThread, Signal

from .state import TrainerState, EpochMetrics


class TrainerRunner(QThread):
    """Lanza un entrenamiento Ultralytics en thread aparte y emite señales por época."""

    epoch_metrics = Signal(object)         # EpochMetrics
    log_line = Signal(str)
    progress = Signal(int, int)             # epoch, total
    finished_ok = Signal(str)               # ruta best.pt
    aborted = Signal()
    failed = Signal(str)

    def __init__(self, state: TrainerState, parent=None):
        super().__init__(parent)
        self.state = state

    def run(self):
        try:
            import torch
            from ultralytics import YOLO
        except Exception as e:
            self.failed.emit(f"No se pudo importar Ultralytics / PyTorch: {e}")
            return

        try:
            st = self.state
            mdl = st.model
            ds = st.dataset
            p = st.params
            a = st.aug

            if not ds.yaml_path or not Path(ds.yaml_path).exists():
                self.failed.emit("Falta data.yaml. Cárgalo en la pestaña Dataset.")
                return

            # Modelo base
            if mdl.custom_weights and Path(mdl.custom_weights).exists():
                weights = str(mdl.custom_weights)
            else:
                weights = mdl.base_weights_name()    # ej. yolov8m.pt

            self.log_line.emit(f"[INFO] Cargando modelo base: {weights}")
            model = YOLO(weights)

            # Run name
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = p.run_name.strip() or f"train_{stamp}"

            proj_dir = Path(__file__).resolve().parents[2] / "runs_train"
            proj_dir.mkdir(parents=True, exist_ok=True)
            st.run_dir = proj_dir / run_name

            # Callback: progreso por época
            def _on_train_epoch_end(trainer):
                try:
                    ep = int(trainer.epoch) + 1
                    metrics = trainer.metrics or {}
                    em = EpochMetrics(
                        epoch=ep,
                        precision=float(metrics.get("metrics/precision(B)", 0.0)),
                        recall=float(metrics.get("metrics/recall(B)", 0.0)),
                        map50=float(metrics.get("metrics/mAP50(B)", 0.0)),
                        map50_95=float(metrics.get("metrics/mAP50-95(B)", 0.0)),
                        box_loss=float(getattr(trainer, "loss_items", [0])[0]) if getattr(trainer, "loss_items", None) is not None else 0.0,
                        lr=float(trainer.optimizer.param_groups[0]["lr"]) if trainer.optimizer else 0.0,
                    )
                    st.history.append(em)
                    if em.map50 > st.best_map50:
                        st.best_map50 = em.map50
                        st.best_epoch = ep
                        st.epochs_no_improve = 0
                    else:
                        st.epochs_no_improve += 1
                    self.epoch_metrics.emit(em)
                    self.progress.emit(ep, p.epochs)
                except Exception as e:
                    self.log_line.emit(f"[WARN] callback error: {e}")

            def _on_train_start(trainer):
                self.log_line.emit(f"[INFO] Entrenamiento iniciado.")
                self.log_line.emit(f"[INFO] Run: {run_name}")
                self.log_line.emit(f"[INFO] Imgsz: {p.imgsz} · Batch: {p.batch} · Epochs: {p.epochs}")

            def _on_train_end(trainer):
                self.log_line.emit(f"[INFO] Entrenamiento finalizado.")

            model.add_callback("on_train_start", _on_train_start)
            model.add_callback("on_train_epoch_end", _on_train_epoch_end)
            model.add_callback("on_train_end", _on_train_end)

            # Cache: ultralytics acepta True/False/'ram'/'disk'
            cache_val: object = p.cache
            if isinstance(cache_val, str) and cache_val.lower() in ("false", "0", "no"):
                cache_val = False

            train_kwargs = dict(
                data=str(ds.yaml_path),
                imgsz=int(p.imgsz),
                epochs=int(p.epochs),
                batch=int(p.batch),
                lr0=float(p.lr0),
                weight_decay=float(p.weight_decay),
                momentum=float(p.momentum),
                warmup_epochs=int(p.warmup_epochs),
                close_mosaic=int(p.close_mosaic),
                label_smoothing=float(p.label_smoothing),
                patience=int(p.patience),
                save_period=int(p.save_period),
                amp=bool(p.amp),
                cos_lr=bool(p.cos_lr),
                cache=cache_val,
                device=p.device,
                workers=int(p.workers),
                # Augmentación
                hsv_h=float(a.hsv_h),
                hsv_s=float(a.hsv_s),
                hsv_v=float(a.hsv_v),
                fliplr=float(a.fliplr),
                mosaic=float(a.mosaic),
                mixup=float(a.mixup),
                copy_paste=float(a.copy_paste),
                # Proyecto
                project=str(proj_dir),
                name=run_name,
                exist_ok=True,
                verbose=True,
                plots=True,
            )

            self.log_line.emit("[INFO] Iniciando model.train(...)")
            results = model.train(**train_kwargs)

            # Best.pt en runs_train/<run_name>/weights/best.pt
            best_path = st.run_dir / "weights" / "best.pt"
            self.finished_ok.emit(str(best_path) if best_path.exists() else "")

        except KeyboardInterrupt:
            self.aborted.emit()
        except Exception as e:
            import traceback
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
