"""Diccionario espanol -> ingles de la interfaz. Solo datos, sin logica.

La clave es la cadena en espanol tal como aparece en el codigo, con sus acentos,
emojis y espacios. Si no coincide exactamente, `tr()` devuelve el espanol y la
traduccion no se aplica: `auditar_traduccion.py` detecta esos casos.

No se traducen: nombres de polimero (PET/PP/LDPE), unidades, siglas de metrica
(mAP, IoU, TP/FP/FN), nombres de archivo ni identificadores de Ultralytics.
"""
from __future__ import annotations

# Cadenas que se dejan igual a proposito: ya estan en ingles, son formatos de Qt,
# nombres propios o identificadores. Se listan para que el auditor no las cuente
# eternamente como pendientes.
NO_TRADUCIR = {
    "Poly-X", "analytics", "bestdetectormedium", "MISCLS", "Alias:",
    "%p%  —  %v / %m", "Device (0/cpu):", "IoU NMS:", "Recall:  —",
    "Poly-X · Detector", "AMP (Mixed Precision FP16)", "FP16 (half precision)",
    "Cache:", "Momentum:", "Patience:", "Workers:", "data.yaml:", "Preset:",
    "Run:", "Device:", "device:", "imgsz:", "conf:", "Label smoothing:",
    "Weight decay:", "Batch size:", "Auto-split", "Close mosaic:",
    "Save period:", "Modelo .pt:", "Pesos .pt:",
}

EN: dict[str, str] = {
    # ══ Barra lateral de los modulos ═══════════════════════════════
    'GT manual': 'Manual GT',
    'Ejecutar': 'Run',
    'Resultados': 'Results',
    'Errores': 'Errors',
    'Comparar': 'Compare',
    'Reporte': 'Report',
    'Augmentación': 'Augmentation',
    'Entrenar': 'Train',
    'Evaluar': 'Evaluate',
    'Exportar': 'Export',
    'Informe': 'Summary',

    # ══ Detector ═══════════════════════════════════════════════════
    "(esperando imagen)": "(waiting for image)",
    "(sin imagen seleccionada)": "(no image selected)",
    "0 cajas": "0 boxes",
    "0 errores": "0 errors",
    "0 imágenes": "0 images",
    "Ajustar (F)": "Fit (F)",
    "Aún no has ejecutado ninguna detección.\nVe a la pestaña Ejecutar e inicia el análisis primero.":
        "You have not run any detection yet.\nGo to the Run tab and start the analysis first.",
    "Aún no se ha ejecutado ninguna corrida.": "No run has been executed yet.",
    "Carga al menos un modelo .pt en la pestaña Modelos.":
        "Load at least one .pt model in the Models tab.",
    "Carga al menos un modelo en la pestaña Modelos.":
        "Load at least one model in the Models tab.",
    "Carpeta GT (opcional):": "Ground-truth folder (optional):",
    "Clase (para nuevas cajas):": "Class (for new boxes):",
    "Clase:": "Class:",
    "Click y arrastra para dibujar · rueda: zoom · Espacio+arrastre: mover · Supr: borrar caja.":
        "Click and drag to draw · wheel: zoom · Space+drag: pan · Del: delete box.",
    "Confianza mín. (conf):": "Min. confidence (conf):",
    "Configura μm/px en Parámetros para ver la distribución de tamaños.":
        "Set µm/px in Parameters to see the size distribution.",
    "Copia esta imagen y sus cajas corregidas a dataset_correcciones/ para mejorar el modelo con fine-tuning (active learning).":
        "Copies this image and its corrected boxes into dataset_correcciones/ so the "
        "model can be improved by fine-tuning (active learning).",
    "Después de ejecutar al menos una detección, presiona <b>Generar reporte HTML</b> (se crea <code>reporte_paper.html</code> dentro de la carpeta del run y se abre en tu navegador) o <b>Exportar a PDF</b> para obtener un archivo listo para enviar. Todas las imágenes van embebidas en base64, por lo que el archivo es autocontenido.":
        "After running at least one detection, press <b>Generate HTML report</b> "
        "(<code>reporte_paper.html</code> is written inside the run folder and opened "
        "in your browser) or <b>Export to PDF</b> for a file ready to send. Every image "
        "is embedded as base64, so the file is self-contained.",
    "Detenido por el usuario.": "Stopped by the user.",
    "Deteniendo…": "Stopping…",
    "Diseñado por\nCristofher Ferrada\n2026": "Designed by\nCristofher Ferrada\n2026",
    "Ejecuta una detección primero.": "Run a detection first.",
    "Error generando PDF": "Error generating PDF",
    "Error generando reporte": "Error generating report",
    "Error guardando": "Error while saving",
    "Falló el renderizado. Como alternativa, genera el HTML y usa\nCtrl+P → 'Guardar como PDF' en tu navegador.":
        "Rendering failed. As an alternative, generate the HTML and use\n"
        "Ctrl+P → 'Save as PDF' in your browser.",
    "Falló la ejecución": "The run failed",
    "Faltan modelos": "Missing models",
    "Genera el reporte y lo guarda como PDF listo para enviar.":
        "Generates the report and saves it as a PDF ready to send.",
    "Generando PDF… (renderizando con el motor del navegador, unos segundos)":
        "Generating PDF… (rendering with the browser engine, a few seconds)",
    "Generando reporte… (puede tardar unos segundos)":
        "Generating report… (this may take a few seconds)",
    "Incluir galería comparativa (Predicción vs Ground Truth)":
        "Include side-by-side gallery (Prediction vs Ground Truth)",
    "Incluir referencias bibliográficas del autor en el reporte":
        "Include the author's bibliographic references in the report",
    "IoU para emparejar Verdaderos Positivos:": "IoU for matching True Positives:",
    "Limpiar todo": "Clear all",
    "Listo. Click y arrastra para dibujar.": "Ready. Click and drag to draw.",
    "Listo. Pulsa «Revisar en pantalla grande» para inspeccionar y corregir.":
        "Ready. Press “Review full screen” to inspect and correct.",
    "Modelo:": "Model:",
    "No encontrado": "Not found",
    "No encontrado en la raíz del proyecto.": "Not found in the project root.",
    "No hay cajas en esta imagen. Corrige o dibuja antes de enviarla.":
        "There are no boxes in this image. Correct or draw some before sending it.",
    "No hay cajas para guardar.": "There are no boxes to save.",
    "No se pudo generar el PDF": "The PDF could not be generated",
    "No se pudo leer el tamaño de la primera imagen.":
        "Could not read the size of the first image.",
    "No se pudo leer la imagen.": "Could not read the image.",
    "No se pudo medir": "Could not measure",
    "PDF no disponible": "PDF not available",
    "Para exportar a PDF se necesita QtWebEngine (incluido en PySide6-Addons).\n\nAlternativa: pulsa 'Generar reporte HTML' y, en el navegador, usa\nCtrl+P → 'Guardar como PDF'.":
        "Exporting to PDF needs QtWebEngine (bundled with PySide6-Addons).\n\n"
        "Alternative: press 'Generate HTML report' and, in the browser, use\n"
        "Ctrl+P → 'Save as PDF'.",
    "Perfil:": "Profile:",
    "Personalizado": "Custom",
    "Precisión:  —": "Precision:  —",
    "Revisión de detecciones — Poly-X": "Detection review — Poly-X",
    "Ruta a la carpeta con .txt YOLO (si no, busca junto a la imagen)":
        "Path to the folder with YOLO .txt files (otherwise it looks next to the image)",
    "Ruta al archivo .pt del modelo entrenado…":
        "Path to the trained model's .pt file…",
    "Selecciona imágenes en la pestaña Imágenes (se usa una para medir).":
        "Select images in the Images tab (one of them is used to measure).",
    "Selecciona imágenes en la pestaña Imágenes.": "Select images in the Images tab.",
    "Selecciona imágenes para ver qué se trocearía.":
        "Select images to see what would be tiled.",
    "Selecciona primero una imagen.": "Select an image first.",
    "Si dejas vacío, busca .txt junto a cada imagen y en /labels/ hermana. Si una imagen tiene GT, se incluirá en el análisis de errores (Verdaderos Positivos, Falsos Positivos y Falsos Negativos). Si no, solo se reportan las detecciones del modelo.":
        "If left empty, .txt files are looked up next to each image and in a sibling "
        "/labels/ folder. An image that has ground truth is included in the error "
        "analysis (True Positives, False Positives and False Negatives); otherwise only "
        "the model's detections are reported.",
    "Si tienes el modelo entrenado por el autor, úsalo como Modelo 1 con un solo clic.":
        "If you have the model trained by the author, use it as Model 1 in one click.",
    "Siguiente  →": "Next  →",
    "Sin ejecución todavía.": "No run yet.",
    "Sin resultados": "No results",
    "Tamaño imagen (imgsz):": "Image size (imgsz):",
    "Tamaño máx (μm):": "Max size (µm):",
    "Tamaño mín (μm):": "Min size (µm):",
    "Tipo:": "Type:",
    "Todos": "All",
    "Usar bestdetectormedium.pt": "Use bestdetectormedium.pt",
    "imgsz máximo": "maximum imgsz",
    "Última imagen procesada por cada modelo:": "Last image processed by each model:",
    "μm por píxel:": "µm per pixel:",
    "←  Anterior": "←  Previous",
    "↶  Deshacer": "↶  Undo",
    "↷  Rehacer": "↷  Redo",
    "⏳ Probando…": "⏳ Testing…",
    "■  Detener": "■  Stop",
    "▶  Iniciar detección": "▶  Start detection",
    "✓  Buena": "✓  Good",
    "✕  Limpiar": "✕  Clear",
    "✗  Mala": "✗  Bad",
    "✗ Falló la generación de PDF.": "✗ PDF generation failed.",
    "✗ Falló la generación.": "✗ Generation failed.",
    "✗ No se pudo generar el PDF.": "✗ The PDF could not be generated.",
    "👁  Revisar en pantalla grande": "👁  Review full screen",
    "💾  Guardar GT (.txt YOLO)": "💾  Save ground truth (.txt YOLO)",
    "💾  Guardar correcciones (.txt YOLO)": "💾  Save corrections (.txt YOLO)",
    "📁  Seleccionar carpeta…": "📁  Select folder…",
    "📂  Abrir carpeta de resultados": "📂  Open results folder",
    "📄  Exportar CSV de detecciones": "📄  Export detections CSV",
    "📄  Generar reporte HTML": "📄  Generate HTML report",
    "📑  Exportar a PDF": "📑  Export to PDF",
    "📤  Enviar al dataset de reentrenamiento": "📤  Send to the retraining dataset",
    "📷  Seleccionar imágenes…": "📷  Select images…",
    "🔍 Detectar máximo (GPU)": "🔍 Detect maximum (GPU)",
    "🗑  Borrar selección": "🗑  Delete selection",

    # ══ Etiquetador ════════════════════════════════════════════════
    "+ Agregar clase": "+ Add class",
    "Anotaciones (imagen actual):": "Annotations (current image):",
    "Carga un modelo .pt primero.": "Load a .pt model first.",
    "Clases activas:": "Active classes:",
    "Conserva el nivel de acercamiento al cambiar de recorte.\nCon cientos de recortes, reencuadrar cada vez cuesta mucho tiempo.\nF reencuadra manualmente.":
        "Keeps the zoom level when moving between crops.\nWith hundreds of crops, "
        "reframing every time costs a lot of time.\nF reframes manually.",
    "Conteo completo": "Counting complete",
    "Deja constancia de que miraste esta imagen aunque no tenga partículas.\nSin esto, una placa vacía revisada es indistinguible de una sin mirar.":
        "Records that you looked at this image even if it has no particles.\nWithout "
        "this, a reviewed empty plate is indistinguishable from one never looked at.",
    "Error al cargar modelo": "Error loading the model",
    "Error en pre-anotación": "Pre-annotation error",
    "Esta imagen ya tiene anotaciones. ¿Sobrescribir con el modelo?":
        "This image already has annotations. Overwrite them with the model?",
    "Mantener zoom entre imágenes": "Keep zoom between images",
    "No quedan imágenes sin revisar.": "There are no unreviewed images left.",
    "Poly-X · Etiquetador": "Poly-X · Labeler",
    "Pre-anotación automática": "Automatic pre-annotation",
    "Pre-anotando": "Pre-annotating",
    "Preset de clases:": "Class preset:",
    "Resolución de inferencia. Con partículas diminutas en fotos grandes,\nun valor bajo no propone nada.":
        "Inference resolution. With tiny particles in large photographs,\na low value "
        "proposes nothing at all.",
    "Salta a la próxima imagen que aún no has revisado (Tab)":
        "Jumps to the next image you have not reviewed yet (Tab)",
    "Siguiente →": "Next →",
    "Sin anotaciones": "No annotations",
    "Sobrescribir": "Overwrite",
    "← Anterior": "← Previous",
    "⏭  Siguiente sin revisar": "⏭  Next unreviewed",
    "✓  Revisada, siguiente   (Espacio)": "✓  Reviewed, next   (Space)",
    "💾  Guardar (.txt)": "💾  Save (.txt)",
    "📂  Abrir carpeta…": "📂  Open folder…",
    "📂  Cargar modelo…": "📂  Load model…",
    "🤖  Pre-anotar TODAS": "🤖  Pre-annotate ALL",
    "🤖  Pre-anotar imagen actual": "🤖  Pre-annotate current image",

    # ══ Entrenador ═════════════════════════════════════════════════
    "%v / %m épocas": "%v / %m epochs",
    "(no se encontró\nla carpeta de train)": "(the train folder\nwas not found)",
    "(sin imágenes\nen train)": "(no images\nin train)",
    "(vacío)": "(empty)",
    "<b>Ninguno</b>: sin transformaciones. &nbsp; <b>Suave</b>: solo flips y jitter HSV. &nbsp; <b>Medio</b> (recomendado): + mosaic, mixup ligero. &nbsp; <b>Fuerte</b>: + copy-paste agresivo.":
        "<b>None</b>: no transformations. &nbsp; <b>Light</b>: flips and HSV jitter only. "
        "&nbsp; <b>Medium</b> (recommended): + mosaic, light mixup. &nbsp; <b>Strong</b>: "
        "+ aggressive copy-paste.",
    "Análisis profundo": "In-depth analysis",
    "Aplica un conjunto de parámetros probados con un clic. Cambiarlo después en 'Parámetros' marca como Personalizado.":
        "Applies a tested set of parameters in one click. Changing it afterwards in "
        "'Parameters' marks it as Custom.",
    "Aún no se han generado runs.": "No runs have been generated yet.",
    "Carga data.yaml en la pestaña Dataset.": "Load data.yaml in the Dataset tab.",
    "Carpeta raíz con images/ y labels/…": "Root folder containing images/ and labels/…",
    "Clases: —": "Classes: —",
    "Cosine LR (curva coseno)": "Cosine LR (cosine schedule)",
    "Estimación de VRAM: —": "VRAM estimate: —",
    "Exportación lista": "Export finished",
    "Falló": "Failed",
    "Falló el entrenamiento": "Training failed",
    "Falta dataset": "Dataset missing",
    "Falta modelo": "Model missing",
    "Familia:": "Family:",
    "Formato:": "Format:",
    "Generando informe…": "Generating report…",
    "Incluir referencias bibliográficas": "Include bibliographic references",
    "Las curvas aparecerán aquí en cuanto el entrenamiento avance la primera época.":
        "The curves appear here as soon as training completes its first epoch.",
    "Learning rate (lr0):": "Learning rate (lr0):",
    "Los porcentajes suman > 100.": "The percentages add up to more than 100.",
    "Nivel:": "Level:",
    "No hay GPU NVIDIA disponible. En CPU recomendamos imgsz ≤ 640.":
        "No NVIDIA GPU available. On CPU we recommend imgsz ≤ 640.",
    "No hay runs disponibles.": "No runs available.",
    "No se encontraron imágenes en images/.": "No images were found in images/.",
    "Nombre del run:": "Run name:",
    "Poly-X · Entrenador": "Poly-X · Trainer",
    "Re-detectar": "Detect again",
    "Ruta a data.yaml…": "Path to data.yaml…",
    "Ruta a un .pt para continuar el entrenamiento…":
        "Path to a .pt file to continue training from…",
    "Selecciona la carpeta raíz primero.": "Select the root folder first.",
    "Selecciona un .pt válido.": "Select a valid .pt file.",
    "Selecciona un data.yaml válido.": "Select a valid data.yaml.",
    "Si tienes un .pt ya entrenado y quieres seguir desde ahí, selecciónalo. Deja vacío para usar el modelo base de la familia/tamaño elegidos.":
        "If you have an already-trained .pt and want to continue from it, select it. "
        "Leave empty to use the base model of the chosen family/size.",
    "Sin GPU no aplica esta sugerencia.": "This suggestion does not apply without a GPU.",
    "Sin data.yaml cargado.": "No data.yaml loaded.",
    "Sin run": "No run",
    "Tamaño:": "Size:",
    "Test %:": "Test %:",
    "Test: —": "Test: —",
    "Toma una carpeta con images/ y labels/ y genera train/val/test automáticamente con un nuevo data.yaml. No mueve archivos: usa listas .txt.":
        "Takes a folder with images/ and labels/ and generates train/val/test "
        "automatically with a new data.yaml. It does not move files: it uses .txt lists.",
    "Train %:": "Train %:",
    "Train: —": "Train: —",
    "Val %:": "Val %:",
    "Val: —": "Val: —",
    "Verifica que el dataset sea correcto antes de entrenar. Se comprueba automáticamente al cargar data.yaml.":
        "Checks that the dataset is sound before training. It runs automatically when "
        "data.yaml is loaded.",
    "Warmup épocas:": "Warmup epochs:",
    "vacío = auto-fecha": "empty = auto date",
    "Épocas:": "Epochs:",
    "• <b>Dataset pequeño (< 500 imgs)</b>: usa Fuerte. Más variedad sintética compensa pocos ejemplos.<br>• <b>Microplásticos PET/PP/LDPE</b>: <b>NO</b> uses HSV-H alto: los colores son la pista principal.<br>• <b>Copy-paste</b> funciona muy bien si el fondo es uniforme (como un filtro).<br>• Si el modelo no converge (mAP no sube), baja el nivel a Suave o Ninguno.":
        "• <b>Small dataset (&lt; 500 imgs)</b>: use Strong. More synthetic variety makes "
        "up for few examples.<br>• <b>PET/PP/LDPE microplastics</b>: do <b>NOT</b> use a "
        "high HSV-H: color is the main cue.<br>• <b>Copy-paste</b> works very well when the "
        "background is uniform (such as a filter).<br>• If the model does not converge "
        "(mAP does not rise), lower the level to Light or None.",
    "• <b>mAP@50</b> debe SUBIR y estabilizarse cerca del 80–95 % para un buen modelo.<br>• <b>Box loss</b> debe BAJAR de forma sostenida. Si oscila o sube, baja el lr0.<br>• Si <b>Sin mejora</b> llega a <b>Patience</b>, el entrenamiento se detendrá solo.":
        "• <b>mAP@50</b> should RISE and settle near 80–95% for a good model.<br>"
        "• <b>Box loss</b> should FALL steadily. If it oscillates or rises, lower lr0.<br>"
        "• If <b>No improvement</b> reaches <b>Patience</b>, training stops on its own.",
    "⌛  Detectando hardware…": "⌛  Detecting hardware…",
    "▶  Iniciar entrenamiento": "▶  Start training",
    "▶  Iniciar validación": "▶  Start validation",
    "✗ Archivo no encontrado.": "✗ File not found.",
    "✗ Error": "✗ Error",
    "🎯  ¿Qué mirar?": "🎯  What to look at",
    "💾  Guardar como…": "💾  Save as…",
    "📂  Abrir carpeta runs_train/": "📂  Open the runs_train/ folder",
    "📤  Exportar": "📤  Export",
    "🔄  Otras 6": "🔄  6 more",
    "🔄  Refrescar lista": "🔄  Refresh list",
    "🔍  Validar ahora": "🔍  Validate now",
    "🪚  Auto-split + generar data.yaml": "🪚  Auto-split + generate data.yaml",

    # ══ Visor ══════════════════════════════════════════════════════
    "  Sin imagen  —  Sin calibración": "  No image  —  Not calibrated",
    "0 detecciones": "0 detections",
    "Abre una imagen primero.": "Open an image first.",
    "Calibración μm/píxel": "µm/pixel calibration",
    "Confianza mínima:": "Minimum confidence:",
    "Error al guardar": "Error while saving",
    "Error en detección": "Detection error",
    "Filtrar por clase:": "Filter by class:",
    "Guardado": "Saved",
    "Haz clic en 2 puntos sobre\nuna referencia conocida.":
        "Click 2 points on\na known reference.",
    "Haz clic en 3 puntos del borde\nde un objeto circular conocido.":
        "Click 3 points on the edge\nof a circular object of known size.",
    "Haz clic en la imagen para marcar\npuntos de referencia.":
        "Click on the image to place\nreference points.",
    "Imagen": "Image",
    "Lado mayor al que se redimensiona la imagen para inferir.\nMás alto = partículas más grandes para la red, pero más memoria.\nSe redondea al múltiplo de 32 más cercano.":
        "Longest side the image is resized to before inference.\nHigher = bigger particles "
        "for the network, but more memory.\nRounded to the nearest multiple of 32.",
    "Muestra las anotaciones YOLO que acompañan a la imagen.\nSirve para revisar el conteo manual sobre la placa, con las\ntallas ya convertidas a µm por la calibración.":
        "Shows the YOLO annotations that come with the image.\nUseful to review the manual "
        "count over the plate, with sizes\nalready converted to µm by the calibration.",
    "No hay imagen abierta.": "No image is open.",
    "Poly-X · Visor": "Poly-X · Viewer",
    "Resolución de inferencia (px):": "Inference resolution (px):",
    "Sin etiquetas": "No labels",
    "Sin imagen": "No image",
    "Todas las clases": "All classes",
    "Usar GPU si está disponible": "Use the GPU if available",
    "▶  Detectar": "▶  Detect",
    "✕  Cancelar calibración": "✕  Cancel calibration",
    "⭕  Círculo": "⭕  Circle",
    "💾  Guardar info actual": "💾  Save current info",
    "📁  Carpeta": "📁  Folder",
    "📄  Cargar etiquetas (.txt)": "📄  Load labels (.txt)",
    "📏  Línea": "📏  Line",
    "📐  Sin calibrar": "📐  Not calibrated",
    "📷  Imagen": "📷  Image",
}
