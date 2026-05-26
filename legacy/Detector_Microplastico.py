import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import pandas as pd
from PIL import Image, ImageTk
from ultralytics import YOLO


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(paths):
    """Acepta lista de archivos o carpetas; devuelve lista final de imágenes."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(IMAGE_EXTS):
                        out.append(os.path.join(root, f))
        else:
            if p.lower().endswith(IMAGE_EXTS):
                out.append(p)
    return sorted(set(out))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def draw_boxes(img_bgr, boxes_xyxy, classes, confs, names, class_filter_ids=None):
    """Dibuja cajas (BGR). class_filter_ids: set de ids permitidos o None."""
    out = img_bgr.copy()
    for (x1, y1, x2, y2), cls, cf in zip(boxes_xyxy, classes, confs):
        cls_i = int(cls)
        if class_filter_ids is not None and cls_i not in class_filter_ids:
            continue

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = f"{names.get(cls_i, str(cls_i))} {cf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


class PolyXApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Poly-X Detector® (Python) — Microplásticos")
        self.geometry("1200x720")

        # Estado global
        self.model = None
        self.model_path = ""
        self.names = {0: "PET", 1: "PP", 2: "LDPE"}  # fallback
        self.selected_polymers = set(["PET", "PP", "LDPE"])  # etiquetas "humanas"
        self.selected_class_ids = None  # set(ids) cuando modelo cargue

        self.conf = tk.DoubleVar(value=0.40)
        self.iou = tk.DoubleVar(value=0.50)
        self.imgsz = tk.IntVar(value=960)

        self.sample_image_path = ""  # para preview sliders
        self.batch_inputs = []       # archivos/carpetas seleccionados
        self.output_dir = os.path.abspath("./runs_detect_polyx")

        # UI: contenedor de pantallas
        self.container = ttk.Frame(self, padding=10)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.frames = {}
        for F in (StartFrame, ModelFrame, PolymerFrame, ParamsFrame, RunFrame):
            frame = F(parent=self.container, app=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show("StartFrame")

    def show(self, name: str):
        self.frames[name].tkraise()

    def log(self, msg: str):
        run = self.frames["RunFrame"]
        run.append_log(msg)


class StartFrame(ttk.Frame):
    def __init__(self, parent, app: PolyXApp):
        super().__init__(parent)
        self.app = app

        title = ttk.Label(self, text="Poly-X Detector®", font=("Segoe UI", 28, "bold"))
        subtitle = ttk.Label(self, text="Detector YOLOv8 para microplásticos (PET / PP / LDPE)",
                             font=("Segoe UI", 12))

        btn = ttk.Button(self, text="Iniciar análisis", command=lambda: app.show("ModelFrame"))

        title.pack(pady=(60, 10))
        subtitle.pack(pady=(0, 30))
        btn.pack(pady=10)


class ModelFrame(ttk.Frame):
    def __init__(self, parent, app: PolyXApp):
        super().__init__(parent)
        self.app = app

        top = ttk.Label(self, text="1) Seleccionar detector (.pt)", font=("Segoe UI", 18, "bold"))
        top.pack(anchor="w", pady=(10, 10))

        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=5)

        ttk.Button(row, text="Cargar best.pt...", command=self.load_model).pack(side=tk.LEFT)
        self.lbl_path = ttk.Label(row, text="(sin modelo cargado)", wraplength=900)
        self.lbl_path.pack(side=tk.LEFT, padx=10)

        info = ttk.LabelFrame(self, text="Propiedades del modelo", padding=10)
        info.pack(fill=tk.X, pady=15)

        self.lbl_classes = ttk.Label(info, text="Clases: -")
        self.lbl_n = ttk.Label(info, text="N° clases: -")
        self.lbl_classes.pack(anchor="w")
        self.lbl_n.pack(anchor="w")

        nav = ttk.Frame(self)
        nav.pack(fill=tk.X, pady=20)
        ttk.Button(nav, text="⬅ Volver", command=lambda: app.show("StartFrame")).pack(side=tk.LEFT)
        ttk.Button(nav, text="Continuar ➜", command=self.next).pack(side=tk.RIGHT)

    def load_model(self):
        p = filedialog.askopenfilename(
            title="Selecciona un modelo YOLO (.pt)",
            filetypes=[("PyTorch weights", "*.pt"), ("Todos", "*.*")]
        )
        if not p:
            return
        self.app.model_path = os.path.abspath(p)
        try:
            self.app.model = YOLO(self.app.model_path)

            # names del modelo
            if hasattr(self.app.model, "names"):
                if isinstance(self.app.model.names, list):
                    self.app.names = {i: n for i, n in enumerate(self.app.model.names)}
                elif isinstance(self.app.model.names, dict):
                    self.app.names = self.app.model.names

            self.lbl_path.config(text=self.app.model_path)
            self.lbl_classes.config(text=f"Clases: {', '.join(self.app.names.values())}")
            self.lbl_n.config(text=f"N° clases: {len(self.app.names)}")

            # Recalcular ids seleccionados (según nombres reales del modelo)
            self._sync_selected_class_ids()

            messagebox.showinfo("Modelo cargado", "Modelo cargado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _sync_selected_class_ids(self):
        # de selected_polymers (strings) -> ids del modelo
        inv = {v: k for k, v in self.app.names.items()}
        ids = set()
        for pol in self.app.selected_polymers:
            if pol in inv:
                ids.add(int(inv[pol]))
        self.app.selected_class_ids = ids if ids else None

    def next(self):
        if self.app.model is None:
            messagebox.showwarning("Falta modelo", "Carga primero un best.pt.")
            return
        self.app.show("PolymerFrame")


class PolymerFrame(ttk.Frame):
    def __init__(self, parent, app: PolyXApp):
        super().__init__(parent)
        self.app = app

        top = ttk.Label(self, text="2) Seleccionar polímeros a detectar", font=("Segoe UI", 18, "bold"))
        top.pack(anchor="w", pady=(10, 10))

        box = ttk.LabelFrame(self, text="Polímeros", padding=10)
        box.pack(fill=tk.X, pady=10)

        self.var_pet = tk.BooleanVar(value=True)
        self.var_pp = tk.BooleanVar(value=True)
        self.var_ldpe = tk.BooleanVar(value=True)

        ttk.Checkbutton(box, text="PET", variable=self.var_pet, command=self._update).pack(anchor="w")
        ttk.Checkbutton(box, text="PP", variable=self.var_pp, command=self._update).pack(anchor="w")
        ttk.Checkbutton(box, text="LDPE", variable=self.var_ldpe, command=self._update).pack(anchor="w")

        self.lbl_state = ttk.Label(self, text="Seleccionado(s): PET, PP, LDPE", font=("Segoe UI", 12))
        self.lbl_state.pack(anchor="w", pady=10)

        nav = ttk.Frame(self)
        nav.pack(fill=tk.X, pady=20)
        ttk.Button(nav, text="⬅ Volver", command=lambda: app.show("ModelFrame")).pack(side=tk.LEFT)
        ttk.Button(nav, text="Continuar ➜", command=self.next).pack(side=tk.RIGHT)

    def _update(self):
        sel = set()
        if self.var_pet.get(): sel.add("PET")
        if self.var_pp.get(): sel.add("PP")
        if self.var_ldpe.get(): sel.add("LDPE")

        if not sel:
            self.lbl_state.config(text="Selecciona al menos 1 polímero.")
        else:
            self.lbl_state.config(text="Seleccionado(s): " + ", ".join(sorted(sel)))

        self.app.selected_polymers = sel

        # Sync ids según nombres del modelo
        inv = {v: k for k, v in self.app.names.items()}
        ids = set()
        for pol in sel:
            if pol in inv:
                ids.add(int(inv[pol]))
        self.app.selected_class_ids = ids if ids else None

    def next(self):
        if not self.app.selected_polymers:
            messagebox.showwarning("Selección vacía", "Selecciona al menos un polímero.")
            return
        self.app.show("ParamsFrame")


class ParamsFrame(ttk.Frame):
    def __init__(self, parent, app: PolyXApp):
        super().__init__(parent)
        self.app = app
        self._debounce_t = 0.0
        self._preview_thread = None

        top = ttk.Label(self, text="3) Ajuste de parámetros (con vista previa)", font=("Segoe UI", 18, "bold"))
        top.pack(anchor="w", pady=(10, 10))

        controls = ttk.Frame(self)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Button(controls, text="Elegir imagen de ejemplo (preview)...", command=self.pick_sample).pack(fill=tk.X, pady=5)

        self.lbl_sample = ttk.Label(controls, text="(sin imagen de ejemplo)", wraplength=320)
        self.lbl_sample.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="Umbral (conf):").pack(anchor="w")
        s1 = ttk.Scale(controls, from_=0.05, to=0.95, variable=self.app.conf, command=lambda _=None: self.schedule_preview())
        s1.pack(fill=tk.X)
        self.lbl_conf = ttk.Label(controls, text="0.40")
        self.lbl_conf.pack(anchor="w", pady=(0, 10))

        ttk.Label(controls, text="NMS (iou):").pack(anchor="w")
        s2 = ttk.Scale(controls, from_=0.05, to=0.95, variable=self.app.iou, command=lambda _=None: self.schedule_preview())
        s2.pack(fill=tk.X)
        self.lbl_iou = ttk.Label(controls, text="0.50")
        self.lbl_iou.pack(anchor="w", pady=(0, 10))

        ttk.Label(controls, text="imgsz:").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.app.imgsz, width=10).pack(anchor="w", pady=(0, 10))
        ttk.Button(controls, text="Actualizar preview", command=self.force_preview).pack(fill=tk.X, pady=5)

        ttk.Separator(controls).pack(fill=tk.X, pady=10)

        nav = ttk.Frame(controls)
        nav.pack(fill=tk.X, pady=10)
        ttk.Button(nav, text="⬅ Volver", command=lambda: app.show("PolymerFrame")).pack(side=tk.LEFT)
        ttk.Button(nav, text="Continuar ➜", command=self.next).pack(side=tk.RIGHT)

        # Preview canvas
        preview_box = ttk.LabelFrame(self, text="Vista previa", padding=10)
        preview_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(preview_box, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._tk_img = None

        self.after(200, self._tick_labels)

    def _tick_labels(self):
        self.lbl_conf.config(text=f"{self.app.conf.get():.2f}")
        self.lbl_iou.config(text=f"{self.app.iou.get():.2f}")
        self.after(200, self._tick_labels)

    def pick_sample(self):
        p = filedialog.askopenfilename(
            title="Selecciona una imagen de ejemplo",
            filetypes=[("Imagen", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos", "*.*")]
        )
        if not p:
            return
        self.app.sample_image_path = os.path.abspath(p)
        self.lbl_sample.config(text=self.app.sample_image_path)
        self.force_preview()

    def schedule_preview(self):
        # debounce simple
        self._debounce_t = time.time()
        self.after(250, self._maybe_preview)

    def _maybe_preview(self):
        if time.time() - self._debounce_t < 0.20:
            return
        self._run_preview_async()

    def force_preview(self):
        self._run_preview_async()

    def _run_preview_async(self):
        if not self.app.sample_image_path:
            return
        if self.app.model is None:
            return
        if self._preview_thread and self._preview_thread.is_alive():
            return

        self._preview_thread = threading.Thread(target=self._preview_worker, daemon=True)
        self._preview_thread.start()

    def _preview_worker(self):
        img_path = self.app.sample_image_path
        conf = float(self.app.conf.get())
        iou = float(self.app.iou.get())
        imgsz = int(self.app.imgsz.get())
        class_ids = self.app.selected_class_ids

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return

        try:
            res = self.app.model.predict(
                source=img_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False
            )[0]

            if res.boxes is None or len(res.boxes) == 0:
                pred = img_bgr
            else:
                boxes = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                pred = draw_boxes(img_bgr, boxes, cls, confs, self.app.names, class_filter_ids=class_ids)

            self.after(0, lambda: self._show(pred))
        except Exception:
            return

    def _show(self, img_bgr):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((cw, ch))
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")

    def next(self):
        self.app.show("RunFrame")


class RunFrame(ttk.Frame):
    def __init__(self, parent, app: PolyXApp):
        super().__init__(parent)
        self.app = app
        self.worker = None

        top = ttk.Label(self, text="4) Ejecutar detección y exportar", font=("Segoe UI", 18, "bold"))
        top.pack(anchor="w", pady=(10, 10))

        controls = ttk.Frame(self)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Button(controls, text="Seleccionar imagen(es)...", command=self.pick_files).pack(fill=tk.X, pady=5)
        ttk.Button(controls, text="Seleccionar carpeta...", command=self.pick_folder).pack(fill=tk.X, pady=5)
        ttk.Button(controls, text="Limpiar selección", command=self.clear_inputs).pack(fill=tk.X, pady=5)

        self.lbl_in = ttk.Label(controls, text="Entrada: (vacía)", wraplength=320)
        self.lbl_in.pack(fill=tk.X, pady=(10, 10))

        ttk.Button(controls, text="Elegir carpeta de salida...", command=self.pick_output).pack(fill=tk.X, pady=5)
        self.lbl_out = ttk.Label(controls, text=f"Salida: {self.app.output_dir}", wraplength=320)
        self.lbl_out.pack(fill=tk.X, pady=(10, 10))

        self.btn_run = ttk.Button(controls, text="▶ Iniciar", command=self.run)
        self.btn_run.pack(fill=tk.X, pady=(10, 5))

        ttk.Button(controls, text="⬅ Volver", command=lambda: app.show("ParamsFrame")).pack(fill=tk.X, pady=5)

        # Preview + Log
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_box = ttk.LabelFrame(right, text="Última imagen procesada", padding=10)
        preview_box.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(preview_box, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._tk_img = None

        log_box = ttk.LabelFrame(right, text="Log", padding=10)
        log_box.pack(fill=tk.BOTH, expand=True)

        self.txt = tk.Text(log_box, height=10, wrap="word")
        self.txt.pack(fill=tk.BOTH, expand=True)

    def append_log(self, s: str):
        self.txt.insert(tk.END, s + "\n")
        self.txt.see(tk.END)

    def pick_files(self):
        files = filedialog.askopenfilenames(
            title="Selecciona una o más imágenes",
            filetypes=[("Imagen", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos", "*.*")]
        )
        if files:
            self.app.batch_inputs.extend(list(files))
            self._refresh_inputs_label()

    def pick_folder(self):
        d = filedialog.askdirectory(title="Selecciona una carpeta con imágenes")
        if d:
            self.app.batch_inputs.append(d)
            self._refresh_inputs_label()

    def clear_inputs(self):
        self.app.batch_inputs = []
        self._refresh_inputs_label()

    def _refresh_inputs_label(self):
        if not self.app.batch_inputs:
            self.lbl_in.config(text="Entrada: (vacía)")
        else:
            self.lbl_in.config(text=f"Entrada: {len(self.app.batch_inputs)} item(s) seleccionado(s)")

    def pick_output(self):
        d = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if d:
            self.app.output_dir = os.path.abspath(d)
            self.lbl_out.config(text=f"Salida: {self.app.output_dir}")

    def run(self):
        if self.app.model is None:
            messagebox.showwarning("Falta modelo", "Carga primero un best.pt.")
            return

        images = list_images(self.app.batch_inputs)
        if not images:
            messagebox.showwarning("Falta entrada", "Selecciona imágenes o carpeta(s) con imágenes.")
            return

        ensure_dir(self.app.output_dir)
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("En curso", "Ya hay una corrida en ejecución.")
            return

        self.btn_run.config(state="disabled")
        self.append_log(f"Procesando {len(images)} imagen(es)...")
        self.worker = threading.Thread(target=self._worker, args=(images,), daemon=True)
        self.worker.start()

    def _worker(self, images):
        conf = float(self.app.conf.get())
        iou = float(self.app.iou.get())
        imgsz = int(self.app.imgsz.get())
        class_ids = self.app.selected_class_ids

        summary_rows = []
        details_rows = []

        for i, img_path in enumerate(images, 1):
            self.append_log(f"[{i}/{len(images)}] {img_path}")

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                self.append_log("  ⚠ No pude leer imagen.")
                continue

            try:
                res = self.app.model.predict(
                    source=img_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    verbose=False
                )[0]
            except Exception as e:
                self.append_log(f"  ❌ Error predict: {e}")
                continue

            names = self.app.names

            if res.boxes is None or len(res.boxes) == 0:
                pred = img_bgr
                counts = {names[k]: 0 for k in names}
                total = 0
            else:
                boxes = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()

                # Filtrar por clases seleccionadas
                keep = []
                for idx in range(len(cls)):
                    cls_i = int(cls[idx])
                    if class_ids is None or cls_i in class_ids:
                        keep.append(idx)

                if keep:
                    boxes_k = boxes[keep]
                    cls_k = cls[keep]
                    confs_k = confs[keep]
                else:
                    boxes_k, cls_k, confs_k = [], [], []

                counts = {names[k]: 0 for k in names}
                for c in cls_k:
                    counts[names[int(c)]] = counts.get(names[int(c)], 0) + 1
                total = int(len(cls_k))

                for (x1, y1, x2, y2), c, cf in zip(boxes_k, cls_k, confs_k):
                    details_rows.append({
                        "image": os.path.basename(img_path),
                        "class_id": int(c),
                        "class_name": names[int(c)],
                        "conf": float(cf),
                        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    })

                pred = draw_boxes(img_bgr, boxes, cls, confs, names, class_filter_ids=class_ids)

            base = os.path.splitext(os.path.basename(img_path))[0]
            out_img = os.path.join(self.app.output_dir, f"{base}_pred.jpg")
            cv2.imwrite(out_img, pred)

            row = {"image": os.path.basename(img_path), "total": total}
            row.update(counts)
            summary_rows.append(row)

            self.after(0, lambda im=pred: self._show(im))

        # Export CSV
        summary_csv = os.path.join(self.app.output_dir, "summary_counts.csv")
        details_csv = os.path.join(self.app.output_dir, "detections_details.csv")
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame(details_rows).to_csv(details_csv, index=False, encoding="utf-8-sig")

        self.append_log("✅ Terminado.")
        self.append_log(f"Salida: {self.app.output_dir}")
        self.append_log(f"- {summary_csv}")
        self.append_log(f"- {details_csv}")

        self.after(0, lambda: self.btn_run.config(state="normal"))
        self.after(0, lambda: messagebox.showinfo("Listo", f"Procesado completado.\nSalida: {self.app.output_dir}"))

    def _show(self, img_bgr):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((cw, ch))
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")


if __name__ == "__main__":
    app = PolyXApp()
    app.mainloop()