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
    "Save period:",
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
    "Después de ejecutar al menos una detección, presiona <b>Generar reporte HTML</b> (se crea <code>informe_deteccion.html</code> dentro de la carpeta del run y se abre en tu navegador) o <b>Exportar a PDF</b> para obtener un archivo listo para enviar. Todas las imágenes van embebidas en base64, por lo que el archivo es autocontenido.":
        "After running at least one detection, press <b>Generate HTML report</b> "
        "(<code>informe_deteccion.html</code> is written inside the run folder and opened "
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

    # ── Añadidas con la comparación de arquitecturas, la selección
    # por dominio, el alcance del informe y el aviso de troceado ──
    'Actualizar': 'Update',
    'Hay una versión nueva disponible en GitHub. Pulsa para descargarla e instalarla.': 'A new version is available on GitHub. Click to download and install it.',
    'Se descargará la versión nueva y se cerrará Poly-X.\n\nTus modelos, resultados y datos no se tocan. ¿Continuar?': 'The new version will be downloaded and Poly-X will close.\n\nYour models, results and data are left untouched. Continue?',
    'No se encontró actualizar.bat en la carpeta de instalación. Descarga la versión nueva manualmente desde GitHub.': 'actualizar.bat was not found in the installation folder. Download the new version manually from GitHub.',
    'Trabajo completo (todas las fotos analizadas)': 'Whole job (every analysed photo)',
    'Solo las fotos que marque abajo': 'Only the photos I tick below',
    'Ambos: un informe completo y otro con las marcadas': 'Both: one complete report and one with the ticked photos',
    'Marcar todas': 'Tick all',
    'Desmarcar todas': 'Untick all',
    'Sin fotos marcadas': 'No photos ticked',
    '⚠ ninguna marcada': '⚠ none ticked',
    '✗ No se pudo generar todo el PDF.': '✗ Could not generate the whole PDF.',
    'Falló el renderizado de: ': 'Rendering failed for: ',
    "Elegiste generar el informe de fotos concretas, pero no hay ninguna marcada.\n\nMarca al menos una en la lista, o cambia el alcance a 'Trabajo completo'.": "You chose a report of specific photos, but none are ticked.\n\nTick at least one in the list, or change the scope to 'Whole job'.",
    'Puedes generar el informe del <b>trabajo completo</b>, solo de las <b>fotos que elijas</b>, o ambos de una vez. Las cifras, los gráficos y la matriz de confusión se recalculan sobre lo que elijas, así que el informe siempre describe exactamente las fotos que muestra.': 'You can generate the report for the <b>whole job</b>, for <b>only the photos you choose</b>, or both at once. Figures, charts and the confusion matrix are recomputed over your choice, so the report always describes exactly the photos it shows.',
    '{} foto(s) en el trabajo': '{} photo(s) in the job',
    '{} de {} marcadas': '{} of {} ticked',
    '{} foto(s) marcada(s)': '{} ticked photo(s)',
    '{} informe(s) generado(s) en {}': '{} report(s) generated in {}',
    '{} PDF generado(s) en {}': '{} PDF(s) generated in {}',
    'Entrenar ambas arquitecturas (v8 y v11) con la misma configuración': 'Train both architectures (v8 and v11) with the same configuration',
    'Elegir el mejor peso según las fotos de terreno, no las de laboratorio': 'Pick the best weight by field photos, not laboratory ones',
    'Se entrenará solo <b>{}</b>. Marca la casilla para entrenar también la otra familia y compararlas en igualdad de condiciones.': 'Only <b>{}</b> will be trained. Tick the box to also train the other family and compare them on equal terms.',
    'Se entrenarán <b>{}</b> uno tras otro, con idénticos imgsz, batch, épocas, semilla y augmentación. Así la diferencia de métricas se puede atribuir a la arquitectura y no a los hiperparámetros.<br>Tarda <b>el doble</b>: van en secuencia porque comparten GPU. Al terminar, la comparación sale en el log y en la pestaña Comparar.': '<b>{}</b> will be trained one after the other, with identical imgsz, batch, epochs, seed and augmentation. That way any difference in metrics can be attributed to the architecture rather than the hyperparameters.<br>It takes <b>twice as long</b>: they run in sequence because they share the GPU. When finished, the comparison appears in the log and in the Compare tab.',
    'Al terminar, se evalúan todos los checkpoints guardados contra <b>solo</b> las imágenes de sedimento real de la validación y se guarda el ganador como <b>best_real.pt</b>.<br>Hace falta porque <code>best.pt</code> lo elige Ultralytics por el mAP global, y en un dataset mixto ese número lo dominan las placas dopadas de laboratorio — el dominio que justamente no transfiere a las fotos de terreno. <code>best.pt</code> se conserva intacto.': 'When training finishes, every saved checkpoint is evaluated against <b>only</b> the real sediment images of the validation split, and the winner is saved as <b>best_real.pt</b>.<br>This is needed because Ultralytics picks <code>best.pt</code> by the global mAP, and in a mixed dataset that number is dominated by the laboratory spiked plates — precisely the domain that does not transfer to field photos. <code>best.pt</code> is left untouched.',
    'Troceado desactivado: cada foto entra completa a imgsz {}. En fotos grandes las partículas pequeñas pueden desaparecer al reescalar.': 'Tiling disabled: each photo enters whole at imgsz {}. In large photos, small particles may vanish when rescaled.',
    'Las fotos ({}×{} px) no llegan al umbral de {} px: se analizan de una pieza.': 'The photos ({}×{} px) do not reach the {} px threshold: they are analysed in one piece.',
    '⚠ Fotos grandes ({}×{} px): se analizarán en {} recortes de {} px con {}% de solape. Va a tardar unas {}× más.<br>El recorte es solo para que el modelo pueda detectar: los resultados y el informe se entregan sobre la foto completa.': '⚠ Large photos ({}×{} px): they will be analysed as {} tiles of {} px with {}% overlap. It will take about {}× longer.<br>Tiling is only so the model can detect: results and the report are delivered on the complete photo.',

    # ── Cadenas que el auditor no veia: viven en asignaciones de
    # clase (PAGE_TITLE = tr(...)) o en llamadas a self.card(tr(...)) ──
    'Poly-X · Suite de microplásticos': 'Poly-X · Microplastics suite',
    'Idioma': 'Language',
    '● PLATAFORMA DE ANÁLISIS': '● ANALYSIS PLATFORM',
    'Plataforma de detección y clasificación de microplásticos': 'Detection and classification platform for microplastics',
    'MÓDULOS': 'MODULES',
    'MÓDULO': 'MODULE',
    'Abrir  →': 'Open  →',
    '📄 LÉAME': '📄 README',
    '📖 Manual de usuario': '📖 User manual',
    'Polímeros': 'Polymers',
    'fluorescencia Nile Red': 'Nile Red fluorescence',
    'medición': 'measurement',
    'calibración por píxel': 'per-pixel calibration',
    'Detector': 'Detector',
    'Entrenador': 'Trainer',
    'Etiquetador': 'Labeler',
    'Visor': 'Viewer',
    'Detección automatizada de PET, PP y LDPE por fluorescencia Nile Red (254 nm) e inteligencia artificial. Entrenamiento, etiquetado, detección y reporte en un mismo flujo.': 'Automated detection of PET, PP and LDPE by Nile Red fluorescence (254 nm) and machine learning. Training, labeling, detection and reporting in a single workflow.',
    '✍  Diseñado y desarrollado por <b>Cristofher Ferrada</b> · Doctorado en Química, 2026': '✍  Designed and developed by <b>Cristofher Ferrada</b> · PhD in Chemistry, 2026',
    'Analiza imágenes con un modelo .pt entrenado. Genera salidas anotadas, CSV con centroides y diámetros, métricas globales e informe HTML de detección.': 'Analyzes images with a trained .pt model. Produces annotated output, a CSV of centroids and diameters, global metrics and an HTML detection report.',
    'Entrena modelos YOLO v8 / v11. Curvas en vivo, recomendaciones automáticas de calidad y comparación con runs anteriores.': 'Trains YOLO v8 / v11 models. Live curves, automatic quality recommendations and comparison against earlier runs.',
    'Anota imágenes en formato YOLO. Soporta pre-anotación con un modelo existente y atajos de teclado, ahorra ~80 % del tiempo manual.': 'Annotates images in YOLO format. Supports pre-annotation with an existing model plus keyboard shortcuts, saving ~80% of the manual time.',
    'Inspección de una imagen a la vez con calibración interactiva μm/píxel (línea o círculo) y medición precisa por partícula.': 'Inspects one image at a time with interactive μm/pixel calibration (line or circle) and precise per-particle measurement.',
    'Sin imágenes': 'No images',
    'Imágenes': 'Images',
    'Modelos': 'Models',
    'Modelo': 'Model',
    'Parámetros': 'Parameters',
    'Sin modelo': 'No model',
    'Error': 'Error',
    'Dataset': 'Dataset',
    '● Listo': '● Ready',
    'Carga rápida': 'Quick load',
    'Carga hasta 3 modelos .pt entrenados para detectar microplásticos. Si cargas más de uno, se compararán automáticamente en el reporte final.': 'Load up to 3 trained .pt models for microplastics detection. If you load more than one, they are compared automatically in the final report.',
    'Origen': 'Source',
    'Imágenes cargadas': 'Loaded images',
    "Selecciona imágenes individuales o una carpeta. Si tienes etiquetas verdaderas en formato YOLO (.txt) en la misma carpeta o en una hermana 'labels/', se cargarán y activarán el análisis de errores.": "Select individual images or a folder. If you have ground truth labels in YOLO format (.txt) in the same folder or in a sibling 'labels/' folder, they are loaded and enable error analysis.",
    'Inferencia': 'Inference',
    'Troceado automático (fotos grandes)': 'Automatic tiling (large photos)',
    'Análisis de errores (si hay GT)': 'Error analysis (when GT is available)',
    'Calibración óptica (tamaño de partícula)': 'Optical calibration (particle size)',
    'Filtro por tamaño (opcional)': 'Size filter (optional)',
    'Cuándo trocear:': 'When to tile:',
    'automático': 'automatic',
    'siempre': 'always',
    'nunca': 'never',
    'Umbral (lado mayor, px):': 'Threshold (longest side, px):',
    'Lado del tile (px):': 'Tile side (px):',
    'Solape entre tiles:': 'Overlap between tiles:',
    'Configura confianza, IoU, calibración óptica y filtros. La calibración μm/px es opcional pero recomendada para reportes científicos.': 'Set confidence, IoU, optical calibration and filters. The μm/px calibration is optional but recommended for scientific reports.',
    'Elige un perfil y listo. «Máxima detección» es el recomendado para microplásticos pequeños en fotos de microscopía de alta resolución.': 'Pick a profile and you are done. «Maximum detection» is recommended for small microplastics in high-resolution microscopy photos.',
    '0.05–0.95. Más alta = menos detecciones (con falsos positivos).': '0.05–0.95. Higher = fewer detections (and fewer false positives).',
    '0.30–0.70. Suprime cajas superpuestas.': '0.30–0.70. Suppresses overlapping boxes.',
    'Estándar COCO: 0.50. Más estricto: 0.75.': 'COCO standard: 0.50. Stricter: 0.75.',
    "'0' = primera GPU. 'cpu' = CPU. '0,1' = GPU 0 y 1.": "'0' = first GPU. 'cpu' = CPU. '0,1' = GPU 0 and 1.",
    '0 = sin filtro. Aplica sobre el diámetro equivalente.': '0 = no filter. Applies to the equivalent diameter.',
    '«auto» decide sola mirando el lado mayor de cada foto.': '«auto» decides on its own from the longest side of each photo.',
    'Más grande = detecta partículas más pequeñas (clave para microplásticos), pero más lento y usa más memoria GPU. Para fotos de alta resolución sube a 4096+. El botón prueba el máximo que aguanta tu GPU con el modelo cargado.': 'Larger = detects smaller particles (key for microplastics), but slower and heavier on GPU memory. For high-resolution photos go up to 4096+. The button probes the maximum your GPU can take with the loaded model.',
    'Una foto de placa completa entra a la red reescalada a imgsz: a 4096 px reducidos a 2080, cada partícula encoge a la mitad y desaparece bajo el stride de la red. Troceando, cada tile entra a resolución nativa y las cajas vuelven a coordenadas de la foto completa, fusionadas con NMS global — no hay que cortar nada a mano ni sumar los conteos después.': 'A full-plate photo enters the network rescaled to imgsz: at 4096 px reduced to 2080, every particle shrinks by half and disappears below the network stride. With tiling, each tile enters at native resolution and the boxes return to full-photo coordinates, merged with a global NMS — nothing has to be cut by hand nor the counts added up afterwards.',
    'Por encima de esto se trocea. 2000 deja pasar enteros los recortes del estudio (1630 px) y trocea las placas completas (~4096 px).': 'Above this, tiling kicks in. 2000 lets the study crops (1630 px) through whole and tiles the full plates (~4096 px).',
    '0 = automático, min(umbral, imgsz). Conviene que coincida con el tamaño de los recortes con que se entrenó el modelo: si el modelo vio partículas a una escala y el tile las presenta a otra, el recall cae.': '0 = automatic, min(threshold, imgsz). It should match the size of the crops the model was trained on: if the model saw particles at one scale and the tile presents them at another, recall drops.',
    '0.25 = cada tile pisa un cuarto del vecino. El solape es lo que evita perder o duplicar la partícula que cae justo en la costura.': '0.25 = each tile overlaps a quarter of its neighbour. The overlap is what avoids losing or duplicating a particle that falls right on the seam.',
    'Verdaderos Positivos: clase correcta + IoU ≥ umbral.  Falsos Positivos: predicción sin GT cercano.  Falsos Negativos: GT no detectado.  Mal Clasificados: IoU alto pero clase distinta.': 'True Positives: correct class + IoU ≥ threshold.  False Positives: prediction with no nearby GT.  False Negatives: GT not detected.  Misclassified: high IoU but wrong class.',
    '0 = sin medición. Ej: objetivo 40× y CMOS calibrado → ~0.244 μm/px.': '0 = no measurement. E.g. 40× objective with a calibrated CMOS → ~0.244 μm/px.',
    'El detector calcula área y diámetro equivalente (de círculo con misma área) para cada partícula detectada. Si hay calibración, también en μm y μm².': 'The detector computes the area and equivalent diameter (of a circle with the same area) for every detected particle. With calibration, also in μm and μm².',
    'Progreso': 'Progress',
    'Preview en vivo': 'Live preview',
    'Inicia la inferencia. Verás progreso en vivo, previews de imágenes anotadas y podrás cancelar en cualquier momento.': 'Starts inference. You will see live progress, previews of annotated images, and you can cancel at any time.',
    'Métricas generales': 'General metrics',
    'Métricas detalladas (modelo principal)': 'Detailed metrics (main model)',
    'Distribución de tamaños (μm) — solo con calibración activa': 'Size distribution (μm) — only with calibration active',
    'Exportar datos': 'Export data',
    'Por imagen': 'Per image',
    'Resumen cuantitativo del análisis. Para gráficos completos y galería por imagen, genera el reporte HTML en la pestaña Reporte.': 'Quantitative summary of the analysis. For full charts and a per-image gallery, generate the HTML report in the Report tab.',
    'Filtros': 'Filters',
    'Cajas con error': 'Boxes with errors',
    'Lista de cajas problemáticas (solo si hay Ground Truth). Falsos Positivos: detección sin GT cercano. Falsos Negativos: GT no detectado. Mal Clasificados: bien localizado, mala clase.': 'List of problematic boxes (only with Ground Truth). False Positives: detection with no nearby GT. False Negatives: GT not detected. Misclassified: located correctly, wrong class.',
    'Resumen por modelo': 'Per-model summary',
    'Detecciones por imagen': 'Detections per image',
    'Si cargaste más de un modelo, aquí ves tabla resumen con todas las métricas lado a lado y cuántas detecciones hizo cada uno por imagen.': 'If you loaded more than one model, here is a summary table with every metric side by side and how many detections each one made per image.',
    'Informe de detección': 'Detection report',
    'Generar informe': 'Generate report',
    'Contenido del reporte': 'Report contents',
    'Alcance del informe': 'Report scope',
    'trabajo completo': 'whole job',
    'Genera un informe HTML autocontenido (todas las imágenes embebidas en base64, así no se rompen al enviarlo a otra persona) con métodos, métricas, gráficos, galería comparativa y análisis de errores. Puedes exportarlo directamente a PDF para enviarlo.': 'Generates a self-contained HTML report (all images embedded in base64, so they do not break when you send it to someone else) with methods, metrics, charts, comparison gallery and error analysis. You can export it straight to PDF.',
    'Anotación': 'Annotation',
    'Anotador completo de Ground Truth. Click izquierdo para dibujar, click sobre una caja para seleccionar, arrastra los handles para redimensionar, Espacio o botón medio para pan, rueda para zoom. Teclas 1..9: clase activa · Supr: borrar · Ctrl+Z/Y: undo/redo · ←/→: imagen anterior/siguiente (autoguarda).': 'Full Ground Truth annotator. Left click to draw, click a box to select it, drag the handles to resize, Space or middle button to pan, wheel to zoom. Keys 1..9: active class · Del: delete · Ctrl+Z/Y: undo/redo · ←/→: previous/next image (autosaves).',
    'Seleccionar modelo': 'Select model',
    'Preset rápido': 'Quick preset',
    'Familia y tamaño': 'Family and size',
    'Pesos personalizados (opcional)': 'Custom weights (optional)',
    "Elige el preset según el caso, o personaliza familia y tamaño. El preset 'Balanceado' es el recomendado para la mayoría de proyectos.": "Pick the preset that fits your case, or customize family and size. The 'Balanced' preset is recommended for most projects.",
    'data.yaml': 'data.yaml',
    'Conteos por split': 'Counts per split',
    'Vista previa (6 imágenes random con cajas)': 'Preview (6 random images with boxes)',
    'Auto-split (si tu dataset no está dividido)': 'Auto-split (if your dataset is not divided)',
    'Validación del dataset': 'Dataset validation',
    'Carga tu data.yaml. Validamos la estructura, contamos imágenes y mostramos una vista previa con cajas. Si tu dataset no está dividido, usa Auto-split.': 'Load your data.yaml. We validate the structure, count images and show a preview with boxes. If your dataset is not divided, use Auto-split.',
    'Parámetros de entrenamiento': 'Training parameters',
    'Configuración básica': 'Basic configuration',
    'Optimizador y learning rate': 'Optimizer and learning rate',
    'Early stopping y checkpoints': 'Early stopping and checkpoints',
    'Hardware e I/O': 'Hardware and I/O',
    'Configuración': 'Configuration',
    '🚀  Maximizar imgsz para mi GPU': '🚀  Maximize imgsz for my GPU',
    'Detectando GPU…': 'Detecting GPU…',
    '🔄  Detectar GPU / VRAM': '🔄  Detect GPU / VRAM',
    '⚡  Sugerir imgsz MÁXIMO': '⚡  Suggest MAXIMUM imgsz',
    'Sugerir batch para mi GPU': 'Suggest batch for my GPU',
    '🎯  Optimizar todo (imgsz → batch → velocidad)': '🎯  Optimize everything (imgsz → batch → speed)',
    'Sin GPU': 'No GPU',
    'Sin GPU utilizable': 'No usable GPU',
    'Configuración optimizada': 'Optimized configuration',
    "Cada parámetro tiene un hint con su explicación. Los valores por defecto son sensatos para microplásticos. Si tocas algo, el preset cambia a 'Personalizado'.": "Every parameter has a hint explaining it. The defaults are sensible for microplastics. If you change anything, the preset switches to 'Custom'.",
    'Fija imgsz al máximo que aguanta la tarjeta (sin pasar de la resolución nativa del dataset), después sube el batch con lo que sobre, y al final ajusta AMP, workers y cache. En ese orden.': "Sets imgsz to the maximum the card can take (without exceeding the dataset's native resolution), then raises the batch with what is left, and finally tunes AMP, workers and cache. In that order.",
    'MÁS ALTO = mejor para partículas pequeñas. Recomendado: 1280+.': 'HIGHER = better for small particles. Recommended: 1280+.',
    'Bajar si hay OOM. Subir si la GPU tiene RAM de sobra.': 'Lower it on OOM. Raise it if the GPU has RAM to spare.',
    '150–300 típico. Early stopping detiene si no mejora (ver Patience).': '150–300 is typical. Early stopping halts if it stops improving (see Patience).',
    'Épocas sin mejora antes de detener. 50 = sensato.': 'Epochs without improvement before stopping. 50 is sensible.',
    'Guardar checkpoint cada N épocas.': 'Save a checkpoint every N epochs.',
    "'0' = primera GPU. 'cpu' = CPU (muy lento).": "'0' = first GPU. 'cpu' = CPU (very slow).",
    "'ram' es lo más rápido si cabe; 'disk' es seguro; 'False' para datasets gigantes.": "'ram' is fastest when it fits; 'disk' is safe; 'False' for huge datasets.",
    'Augmentación de datos': 'Data augmentation',
    'Nivel de augmentación': 'Augmentation level',
    'Sliders manuales (avanzado)': 'Manual sliders (advanced)',
    'Recomendaciones': 'Recommendations',
    'Aumenta artificialmente la variedad del dataset. Útil para datasets pequeños (< 500 imágenes). Demasiado aug = el modelo no converge; muy poco = sobreajuste.': 'Artificially increases the variety of the dataset. Useful for small datasets (< 500 images). Too much augmentation and the model will not converge; too little and it overfits.',
    'Control': 'Control',
    'Inicia el entrenamiento del modelo con los parámetros configurados. Verás curvas en vivo, métricas y log. Puedes detenerlo en cualquier momento.': 'Starts training the model with the configured parameters. You will see live curves, metrics and the log. You can stop it at any time.',
    'Valida cualquier modelo .pt sobre un dataset YOLO (data.yaml). Útil para comprobar la calidad de un best.pt sobre un test set distinto al de entrenamiento.': 'Validates any .pt model against a YOLO dataset (data.yaml). Useful for checking the quality of a best.pt on a test set other than the training one.',
    'Comparar runs': 'Compare runs',
    'Runs disponibles': 'Available runs',
    'Tabla con todos los entrenamientos anteriores en runs_train/. Útil para comparar configuraciones y elegir el mejor modelo.': 'A table of every previous training run in runs_train/. Useful for comparing configurations and picking the best model.',
    'Exportar modelo': 'Export model',
    'Modelo a exportar': 'Model to export',
    'Formato y opciones': 'Format and options',
    'Convierte el modelo entrenado a otros formatos para producción: ONNX (portable), TensorRT (NVIDIA rápido), TFLite (móvil), CoreML (Apple), etc.': 'Converts the trained model to other production formats: ONNX (portable), TensorRT (fast on NVIDIA), TFLite (mobile), CoreML (Apple), and so on.',
    'Informe del entrenamiento': 'Training report',
    'Selección de run': 'Run selection',
    'Generar': 'Generate',
    'Genera un informe HTML autocontenido con las curvas y métricas del run elegido. Listo para convertir a PDF (Ctrl+P en el navegador).': 'Generates a self-contained HTML report with the curves and metrics of the chosen run. Ready to convert to PDF (Ctrl+P in the browser).',

    # ── Descripciones de tamano del modelo (SIZE_DESCRIPTIONS) ──
    'El más liviano y rápido. Útil para probar.': 'The lightest and fastest. Useful for testing.',
    'Buen balance velocidad/precisión.': 'A good speed/accuracy balance.',
    'RECOMENDADO. Precisión sólida en GPU media.': 'RECOMMENDED. Solid accuracy on a mid-range GPU.',
    'Más preciso, requiere GPU potente.': 'More accurate, needs a powerful GPU.',
    'El más preciso, muy lento sin GPU buena.': 'The most accurate, very slow without a good GPU.',
    # ── Anadidas en agosto de 2026: calibracion contra la placa, medida de
    # forma de particula, selector de secciones del informe y revision
    # particula a particula en el Visor. ──
    '   ⚠ Vuelve a Ejecutar la detección: el informe usa el GT leído en la última corrida.':
        '   ⚠ Run the detection again: the report uses the ground truth read during the last run.',
    'No hay ninguna caja dibujada. ¿Guardar esta imagen como revisada con cero partículas? Se sobrescribirá la anotación anterior, si la hubiera.':
        'No boxes have been drawn. Save this image as reviewed with zero particles? Any previous annotation will be overwritten.',
    'Medir la placa en cada foto y usar su escala (recomendado)':
        'Measure the dish in every photo and use its scale (recommended)',
    'Respaldo: se usa solo si no se puede medir la placa. 0 = sin medición.':
        'Fallback: used only when the dish cannot be measured. 0 = no calibration.',
    'Diámetro real de la placa (mm):':
        'Actual dish diameter (mm):',
    'Diámetro externo nominal. La placa Petri del estudio mide 100 mm.':
        'Nominal outer diameter. The Petri dish used in this study is 100 mm.',
    'Altura de la placa (mm):':
        'Dish height (mm):',
    '0 = no corregir. Una Petri de 100 mm suele medir 15 mm de alto.':
        '0 = no correction. A 100 mm Petri dish is usually 15 mm tall.',
    'Distancia cámara → base (mm):':
        'Camera-to-base distance (mm):',
    '0 = no corregir. Medir hasta el FONDO de la placa, no hasta el borde.':
        '0 = no correction. Measure to the BOTTOM of the dish, not to its rim.',
    'No hay que marcar nada: el borde de la placa se encuentra solo. El radio se obtiene ajustando un círculo al anillo muestreado en 720 direcciones, con rechazo de atípicos; Hough solo aporta el centro aproximado porque su radio llega a errar un 12 %. En los recortes la placa no aparece, así que ahí la escala se hereda del índice de calibración que dejó el recorte.':
        'Nothing has to be marked: the dish rim is found automatically. The radius comes from fitting a circle to the rim sampled along 720 directions, with outlier rejection; the Hough transform only supplies the approximate centre, because its radius can be off by as much as 12 %. In crops the dish is not visible, so there the scale is inherited from the calibration index left behind when the crop was made.',
    'El borde de la placa está más cerca de la cámara que el fondo donde reposan las partículas, de modo que se proyecta más grande y la escala sale pequeña: sin corregir, TODAS las tallas quedan subestimadas. Con la placa a 15 mm y la cámara a 100 mm son casi 16 puntos porcentuales; a 500 mm, un 3 %. Rellena los dos campos de arriba para corregirlo.':
        'The dish rim sits closer to the camera than the base where the particles lie, so it projects larger and the resulting scale is too small: left uncorrected, ALL sizes are underestimated. With a 15 mm dish and the camera at 100 mm that is almost 16 percentage points; at 500 mm, about 3 %. Fill in both fields above to correct it.',
    'Genera un informe HTML autocontenido (todas las imágenes embebidas en base64, así no se rompen al enviarlo a otra persona) con métodos, métricas, gráficos, galería comparativa, conteo por muestra y análisis de errores. Puedes exportarlo directamente a PDF para enviarlo, o guardar aparte cada foto con sus etiquetas dibujadas.':
        'Generates a self-contained HTML report (every image embedded as base64, so nothing breaks when you send it to someone else) with methods, metrics, charts, a comparison gallery, per-sample counts and error analysis. You can export it straight to PDF to send it, or save each photo separately with its labels drawn on.',
    '📄  Generar informe HTML':
        '📄  Generate HTML report',
    'Guardar las fotos con las etiquetas':
        'Save the photos with their labels',
    'Guarda cada foto analizada con sus cajas dibujadas, en su resolución original. Eliges una carpeta y dentro se crea una subcarpeta por cada opción que marques (<code>conteo_manual</code>, <code>deteccion_modelo</code>, <code>ambas_superpuestas</code>), así no se mezclan. Respeta el <b>alcance</b> elegido arriba: si marcaste fotos concretas, solo se guardan esas.':
        'Saves every analysed photo with its boxes drawn on, at its original resolution. You pick a folder and a subfolder is created inside it for each option you tick (<code>conteo_manual</code>, <code>deteccion_modelo</code>, <code>ambas_superpuestas</code>), so they do not get mixed up. It honours the <b>scope</b> chosen above: if you ticked specific photos, only those are saved.',
    'Etiquetas manuales (Ground Truth)':
        'Manual labels (ground truth)',
    'Detecciones del modelo':
        'Model detections',
    'Las dos superpuestas en la misma foto':
        'Both overlaid on the same photo',
    '🖼️  Guardar fotos etiquetadas':
        '🖼️  Save labelled photos',
    'Carpeta donde guardar las fotos etiquetadas':
        'Folder to save the labelled photos in',
    'Qué incluir en el informe':
        'What to include in the report',
    'El informe completo es largo. Desmarca lo que no necesites: las secciones se renumeran solas y el índice se ajusta. Una sección marcada que no tenga datos —errores sin ground truth, por ejemplo— se omite igualmente.':
        'The full report is long. Untick whatever you do not need: sections renumber themselves and the table of contents adjusts. A ticked section with no data — error analysis without ground truth, for instance — is left out anyway.',
    'Galería por imagen  (es lo que más pesa)':
        'Per-image gallery  (this is what makes the file big)',
    'Nada que guardar':
        'Nothing to save',
    'Marca al menos una de las tres opciones: etiquetas manuales, detecciones del modelo, o las dos superpuestas.':
        'Tick at least one of the three options: manual labels, model detections, or both overlaid.',
    "El alcance está en 'solo las fotos que marque' y no hay ninguna marcada.":
        "The scope is set to 'only the photos I tick' and none are ticked.",
    "No se encontró ninguna imagen anotada en disco para lo que pediste.\n\nSi marcaste 'etiquetas manuales', recuerda que solo existen para las fotos que tienen Ground Truth.":
        "No annotated image was found on disk for what you asked for.\n\nIf you ticked 'manual labels', remember that these only exist for photos that have ground truth.",
    'Completo':
        'Full',
    'Resumen breve':
        'Short summary',
    'Metodológico':
        'Methodological',
    'Guardando {} imagen(es)…':
        'Saving {} image(s)…',
    'Algunas no se pudieron guardar':
        'Some could not be saved',
    '{} imagen(es) en {} carpeta(s) dentro de {}':
        '{} image(s) in {} folder(s) inside {}',
    '{} fallaron':
        '{} failed',
    'Métricas detalladas (suma de todos los modelos)':
        'Detailed metrics (summed over all models)',
    'Ground truth':
        'Ground truth',
    'No pide ninguna carpeta: relee los .txt que están junto a cada foto, los mismos que ves en GT manual. Hace falta porque la corrida guarda el ground truth que leyó, así que corregir una anotación después no cambia por sí solo estas métricas ni el informe. Recalcula sin pasar el modelo otra vez y solo redibuja las imágenes cuyo .txt cambió.':
        'It asks for no folder: it re-reads the .txt files sitting next to each photo, the same ones you see under Manual GT. This is needed because a run freezes the ground truth it read, so fixing an annotation afterwards does not by itself change these metrics or the report. It recalculates without running the model again, and only redraws the images whose .txt actually changed.',
    '🔄  Releer los .txt del disco y recalcular':
        '🔄  Re-read the .txt files from disk and recalculate',
    '🔍  Revisar partícula a partícula en el Visor':
        '🔍  Review particle by particle in the Viewer',
    'Abre la imagen seleccionada en el Visor con estas mismas detecciones, numeradas. Allí cada partícula muestra sobre qué se midió y si es fibra o fragmento — sin volver a pasar el modelo.':
        'Opens the selected image in the Viewer with these very detections, numbered. There each particle shows what it was measured on and whether it is a fibre or a fragment — without running the model again.',
    'Releyendo ground truth…':
        'Re-reading ground truth…',
    'Cancelar':
        'Cancel',
    'Recargar GT':
        'Reload GT',
    'Revisar':
        'Review',
    'Selecciona primero una fila de la tabla de abajo.':
        'Select a row from the table below first.',
    'Esa imagen no tiene detecciones que revisar.':
        'That image has no detections to review.',
    'Guardar también el mejor peso para el conjunto real':
        'Also save the best weights for the real dataset',
    'Al terminar quedan tres pesos en la carpeta del run: <b>best_sintetico.pt</b>, <b>best_real.pt</b> y <b>last.pt</b> (este último sirve para reanudar el entrenamiento).':
        'When it finishes there are three sets of weights in the run folder: <b>best_sintetico.pt</b>, <b>best_real.pt</b> and <b>last.pt</b> (this last one is what you use to resume training).',
    '📁  Cargar predicciones de una corrida':
        '📁  Load predictions from a run',
    'Lee las predicciones que el Detector dejó en runs/detect_.../ para esta misma foto. Sirve para revisar una corrida ya cerrada, sin volver a pasar el modelo.':
        'Reads the predictions the Detector left in runs/detect_.../ for this same photo. Use it to review a run you have already closed, without running the model again.',
    'Selecciona una partícula para ver su medición':
        'Select a particle to see how it was measured',
    'Carpeta de la corrida (runs/detect_...)':
        'Run folder (runs/detect_...)',
    'Abre primero la foto original.':
        'Open the original photo first.',
    'Sin predicciones':
        'No predictions',
    'Varios modelos':
        'Several models',
    'Esta foto se analizó con varios modelos. ¿Cuál revisar?':
        'This photo was analysed with several models. Which one do you want to review?',
    'El archivo está vacío: esa foto no tuvo detecciones.':
        'The file is empty: that photo had no detections.',
    'No se pudo aislar esta partícula del fondo':
        'This particle could not be separated from the background',
    '● {n} predicciones cargadas de {c}':
        '● {n} predictions loaded from {c}',
    "En esa corrida no hay predicciones para {n}.\n\nSe buscó en las subcarpetas 'labels'.":
        "That run has no predictions for {n}.\n\nThe 'labels' subfolders were searched.",
    # ── Cadenas que estaban fuera de tr() y se detectaron arrancando los
    # tres modulos en ingles y buscando texto con aspecto de espanol. ──
    'Detecciones': 'Detections',
    'Confianza media': 'Mean confidence',
    'Tamaño medio (μm)': 'Mean size (μm)',
    'Época actual': 'Current epoch',
    'Sin mejora': 'No improvement',
    'Split train con imágenes': 'Train split with images',
    'Split val con imágenes': 'Val split with images',
    'Resumen': 'Summary',
    'Métodos': 'Methods',
    'Calibración de escala': 'Scale calibration',
    'Forma y talla de las partículas': 'Particle shape and size',
    'Ficha de partículas medidas (muestra)': 'Measured-particle cards (sample)',
    'Resultados generales': 'Overall results',
    'Análisis de errores': 'Error analysis',
    'Comparación entre modelos': 'Model comparison',
    'Galería por imagen': 'Per-image gallery',
    'Conteo por muestra y tipo de plástico': 'Counts by sample and polymer type',
    'Referencias bibliográficas': 'References',
    # El indice del informe y el aviso de deteccion de GPU: las dos ultimas
    # cadenas visibles que quedaban sin traducir.
    '<ol><li><b>Resumen</b> con detecciones, confianza media y tamaño medio.</li><li><b>Métodos</b> — modelo, parámetros, calibración y dispositivo.</li><li><b>Calibración de escala</b> — de dónde salió el µm/px de cada foto.</li><li><b>Forma y talla</b> — reparto fibra/fragmento, distribución de tallas, la mayor y la menor, recuento por imagen y cómo se mide el largo.</li><li><b>Fichas</b> — una muestra de partículas con su medición dibujada.</li><li><b>Resultados generales</b> — clases, confianza y tamaños.</li><li><b>Resumen por modelo</b> (tabla comparativa)</li><li><b>Análisis de errores</b> (solo si hay ground truth) — matriz de confusión y P/R/F1 por clase.</li><li><b>Comparación entre modelos</b></li><li><b>Galería por imagen</b> — predicción y ground truth lado a lado.</li><li><b>Conteo por muestra y tipo de plástico</b> — por imagen, por tramo y por estación.</li><li><b>Referencias bibliográficas</b></li></ol><p>Todas las secciones son opcionales: se eligen arriba.</p>':
        "<ol><li><b>Summary</b> with detections, mean confidence and mean size.</li><li><b>Methods</b> — model, parameters, calibration and device.</li><li><b>Scale calibration</b> — where each photo's µm/px came from.</li><li><b>Shape and size</b> — fibre/fragment split, size distribution, the largest and the smallest, per-image counts and how length is measured.</li><li><b>Particle cards</b> — a sample of particles with their measurement drawn on.</li><li><b>Overall results</b> — classes, confidence and sizes.</li><li><b>Per-model summary</b> (comparison table)</li><li><b>Error analysis</b> (only when ground truth is available) — confusion matrix and per-class P/R/F1.</li><li><b>Model comparison</b></li><li><b>Per-image gallery</b> — prediction and ground truth side by side.</li><li><b>Counts by sample and polymer type</b> — per image, per depth interval and per station.</li><li><b>References</b></li></ol><p>Every section is optional: you choose them above.</p>",
    '⌛  Detectando GPU en background (puede tardar 10–30 s la primera vez)…':
        '⌛  Detecting GPU in the background (may take 10–30 s the first time)…',
    # ── Ultimo barrido: alias por defecto, estimacion de VRAM y
    # descripciones de los formatos de exportacion. ──
    'Modelo .pt:': 'Model .pt:',
    'Pesos .pt:': 'Weights .pt:',
    'Modelo {n}': 'Model {n}',
    'Estimación de VRAM: ~{gb} (modelo {size}, imgsz {imgsz}, batch {batch}, AMP={amp})': 'VRAM estimate: ~{gb} (model {size}, imgsz {imgsz}, batch {batch}, AMP={amp})',
    'Más portable. Para CPU o GPU vía onnxruntime.': 'Most portable. For CPU or GPU via onnxruntime.',
    'Modelo nativo de PyTorch.': 'Native PyTorch model.',
    'Solo NVIDIA. Inferencia más rápida en producción.': 'NVIDIA only. Fastest inference in production.',
    'Intel CPU/iGPU.': 'Intel CPU/iGPU.',
    'Mobile/Edge devices.': 'Mobile/edge devices.',
    'Apple Silicon (M1/M2/M3) y iOS.': 'Apple Silicon (M1/M2/M3) and iOS.',
}
