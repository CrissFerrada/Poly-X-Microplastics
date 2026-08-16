# Poly-X — Suite de detección de microplásticos

***Español** · [English](README.en.md)*

> **Detección automatizada y clasificación de microplásticos (PET, PP, LDPE)  
> por fluorescencia Nile Red bajo luz UV (254 nm) con modelos YOLO v8/v11.**

**Autor:** Cristofher Ferrada · Doctorado en Ciencias mención Química · PUCV · 2026  
**Versión:** 2.0.0 · Windows 10/11 · Python 3.11

---

## Módulos

| # | Módulo | Estado | Descripción |
|---|---|---|---|
| 1 | 🏠 **Launcher** | ✅ Funcional | Menú principal — lanza cada módulo en ventana independiente |
| 2 | 🔬 **Detector** | ✅ Funcional | Análisis en lote con YOLO · reporte HTML paper-quality |
| 3 | 🎯 **Entrenador** | ✅ Funcional | Entrenamiento YOLO v8/v11 con curvas en vivo |
| 4 | 🏷 **Etiquetador** | ✅ Funcional | Anotación YOLO interactiva con pre-anotación automática |
| 5 | 📐 **Visor** | ✅ Funcional | Inspección de imágenes con calibración interactiva μm/píxel |

---

## Características

### 🔬 Detector (análisis en lote)
- Carga hasta **3 modelos .pt** en paralelo para comparación directa
- Inferencia sobre carpetas de imágenes con barra de progreso
- Análisis de errores: **Verdaderos Positivos / Falsos Positivos / Falsos Negativos / Mal Clasificados**, matriz de confusión, galería de falsos positivos/negativos
- Calibración **μm/px** para reportar tamaños reales de partículas
- **Histograma de distribución de tamaños** por clase (PET/PP/LDPE) — visible cuando hay calibración activa
- **Exportar CSV** directo desde la página de resultados
- **Drag & Drop** de modelos `.pt` e imágenes/carpetas
- **Resolución de inferencia** ajustable con perfiles (Rápido 1280 · Equilibrado
  2560 · Máxima detección 4096) y un botón que mide el máximo que aguanta tu GPU
- Reporte HTML autocontenido (imágenes incrustadas en **base64**, calidad paper)
  con **galería comparativa Predicción vs Ground Truth (lado a lado)**. Las
  imágenes de galería se recodifican y se limita su número para que el archivo
  siga siendo abrible en un navegador; las métricas cubren todas las imágenes
- **Exportación a PDF** del reporte (un clic), listo para enviar por correo
- **Alcance del informe elegible**: trabajo completo, solo las fotos que marques, o
  ambos de una vez. Las cifras, los gráficos y la matriz de confusión se recalculan
  sobre lo elegido, así que el informe siempre describe las fotos que muestra
- **Comparación real entre modelos** dentro del informe: tabla foto por foto con las
  detecciones de cada uno y, si hay ground truth, sus TP/FP/FN y el F1 global
- **Troceado automático de fotos grandes**: por encima del umbral la foto se analiza
  en recortes solapados, porque a resolución completa las partículas caen por debajo
  del stride de la red y desaparecen. Las cajas vuelven a coordenadas de la foto
  original y los solapes se fusionan con NMS, de modo que **los resultados y el
  informe se entregan siempre sobre la foto completa, nunca sobre los recortes**
- Aviso previo en la pestaña Ejecutar de si el lote se va a trocear y cuánto tardará

### 🎯 Entrenador
- Soporte para **YOLO v8 y v11**, tamaños nano → xlarge
- Carga de `data.yaml` con auto-detección de splits train/val/test
- **Validación del dataset** con indicadores ✓/✗ (imágenes, labels, clases)
- Curvas de pérdida en vivo (Box loss, mAP, Precision, Recall)
- Augmentación configurable (flip, rotation, mosaic, HSV)
- Exportación a **ONNX / TensorRT / CoreML**
- Diagnóstico explícito de GPU/CPU al iniciar el entrenamiento
- **Entrenar v8 y v11 con la misma configuración** desde una casilla: ambas corridas
  reutilizan idénticos `imgsz`, `batch`, épocas, semilla y augmentación, que es lo que
  permite atribuir la diferencia de métricas a la arquitectura y no a los
  hiperparámetros. Van en secuencia porque comparten GPU, y al terminar el log resume
  la comparación
- **Auditoría del dataset** antes de entrenar: distribución de clases, formato y
  validación, con veredicto de si es apto

### 🏷 Etiquetador
- Canvas interactivo: **arrastrar** para dibujar cajas YOLO normalizadas
- Clic derecho para asignar clase; teclas `1`–`9` para cambiar clase activa
- **Undo/Redo** (Ctrl+Z / Ctrl+Y) por imagen
- **Seguimiento del avance**, pensado para campañas de cientos de imágenes
  repartidas en varias sesiones. El estado se recupera del disco al abrir la
  carpeta y la lista distingue tres casos:

  | Marca | Significado |
  |---|---|
  | `✓ nombre (n)` | revisada, con n objetos |
  | `· nombre (0)` | revisada, sin objetos — **es un dato** |
  | `○ nombre` | todavía sin revisar |

  Un `.txt` solo se escribe al marcar la imagen como revisada o al dibujar una
  caja: pasar de largo no crea archivo, porque registrar como «revisada con
  cero» una imagen apenas ojeada falsearía un conteo censal.
- `Espacio` marca revisada y avanza · `Tab` salta a la siguiente sin revisar ·
  `F` reencuadra · `←/→` navega · `Supr` borra la caja seleccionada
- **El zoom se conserva** entre imágenes (casilla en el panel derecho)
- **Lado mínimo de caja: 2 px.** Al rechazar una caja se avisa en la barra de
  estado, en vez de descartarla en silencio
- **Pre-anotación automática** con modelo `.pt`, con `conf` e `imgsz` ajustables
  y GPU si está disponible
- **Auto-guardado silencioso** cada 60 segundos; miniaturas generadas en segundo
  plano para que la ventana responda de inmediato
- Guarda `.txt` YOLO junto a la imagen (o en `labels/` si está en `images/`) y
  genera `classes.txt` en la raíz y en cada subcarpeta

### 📐 Visor
- Abre una imagen individual o navega una carpeta con `← →`
- **Calibración interactiva μm/píxel** en 2 modos:
  - 📏 **Línea** (2 clics): marca una referencia conocida, ingresa su tamaño real
  - ⭕ **Círculo** (3 clics): marca 3 puntos del borde, ingresa el diámetro real
    (útil con placas Petri, cuyo diámetro es conocido)
- Barra inferior muestra `📐 0.4880 μm/px (línea)` en tiempo real
- Detección con modelo `.pt` cargado, con **resolución de inferencia
  configurable** (320–8192) y GPU si está disponible. Con objetos diminutos en
  fotos de alta resolución el `imgsz` es determinante: a valores bajos las
  partículas caen por debajo del *stride* de la red y no se detecta nada
- **Cargar etiquetas `.txt`**: muestra anotaciones ya existentes sobre la imagen,
  con sus tallas convertidas a μm. Sirve para revisar un conteo manual sin
  volver al Etiquetador
- Tabla de resultados filtrable por clase con Ø(px) y Ø(μm)
- **Drag & Drop** de imagen o modelo `.pt` directo al canvas
- Exporta: imagen anotada + `detecciones.csv` + `resumen.json`

---

## Polímeros detectados

| ID | Clase | Fluorescencia observada (Nile Red, UV) | Color en la interfaz |
|---|---|---|---|
| 0 | **PET** | Rojo–salmón | 🔴 `#e3342f` |
| 1 | **PP** | **Amarillo verdoso, apagado** | 🟠 `#ff8c00` |
| 2 | **LDPE** | Amarillo franco y **más brillante** | 🟡 `#ffd700` |

Los colores de la derecha son solo los de las cajas en pantalla; no describen la
emisión real. Medido sobre el interior de las cajas del dataset de entrenamiento
(media RGB, n=30 por clase):

| Clase | R | G | B |
|---|---|---|---|
| PET | 116 | 58 | 65 |
| PP | 122 | **125** | 32 |
| LDPE | 181 | 162 | 57 |

**PP y LDPE no se distinguen por tono sino por brillo**: ambos son amarillentos,
pero en PP el verde iguala o supera al rojo y la emisión es bastante más apagada.
Es la confusión más habitual al anotar, y la razón de que el recall por clase caiga
en esas dos.

---

## Idioma

La interfaz está en **español e inglés**. El selector está en la barra superior
del Launcher y la elección se recuerda entre sesiones. La primera vez se toma el
idioma del sistema; `POLYX_IDIOMA=en` lo fuerza sin tocar la interfaz.

Los módulos son procesos aparte y leen el idioma al abrirse, así que el cambio
surte efecto en cuanto abras el siguiente módulo.

Para ver qué falta por traducir:

```bat
.venv\Scripts\python.exe auditar_traduccion.py
```

---

## Requisitos

| Componente | Versión |
|---|---|
| Windows | 10 / 11 |
| Python | **3.11.x** (no 3.12+) |
| RAM | 8 GB mínimo |
| GPU | NVIDIA opcional (20–30× más rápido para entrenamiento). El instalador elige la versión de CUDA según la arquitectura de la tarjeta |

Dependencias principales: `PySide6 6.7`, `Ultralytics 8.3`, `OpenCV 4.10`, `NumPy 1.26`, `Matplotlib 3.9`

---

## Instalación

Tras descargar el proyecto desde GitHub (botón **Code → Download ZIP**, y descomprimir;
o `git clone`), haz **doble clic en**:

```bat
SETUP.bat
```

El instalador:

1. **Pregunta dónde instalar Poly-X** (pulsa ENTER para instalarlo en la misma carpeta, o escribe otra ruta).
2. Detecta Python 3.11 y crea el entorno `.venv`.
3. Detecta la GPU y descarga PyTorch con **la versión de CUDA que corresponde a esa
   tarjeta**. Las RTX 50xx (Blackwell, `sm_120`) necesitan CUDA 12.8: con las ruedas
   de CUDA 11.8 `torch.cuda.is_available()` devuelve `True` y el fallo aparece recién
   al entrenar, con *"no kernel image is available for execution on the device"*.
4. Instala el resto de dependencias, verifica que todo importa y **comprueba que
   PyTorch traiga kernels para tu GPU concreta**.
5. Registra la versión instalada, para que el launcher pueda avisarte de las nuevas.
6. **Busca instalaciones anteriores de Poly-X** en el equipo. Si encuentra alguna,
   muestra qué datos tiene, los migra a la nueva y manda la vieja a la **papelera**
   (ver más abajo).
7. Te ofrece **crear un acceso directo "Poly-X" en el Escritorio** y **te dice cómo iniciarlo**.

### Abrir sin consola

Doble clic en **`Poly-X.vbs`** (raíz del proyecto): abre el Launcher directamente,
sin ventana negra. Resuelve su propia ubicación, así que funciona desde cualquier
carpeta — basta descargar el repositorio y hacer doble clic, sin editar rutas.

Para tenerlo a mano, clic derecho sobre `Poly-X.vbs` → *Enviar a* → *Escritorio
(crear acceso directo)*, y asígnale el icono `assets\polyx.ico`.

`iniciar_polyx.bat` hace lo mismo pero mostrando la consola, útil para ver
mensajes de error si algo falla al arrancar.

> **Importante:** los modelos entrenados (`*.pt`) **no** se incluyen en la descarga de GitHub
> por su tamaño. Copia tu archivo `.pt` dentro de la carpeta `models\` para usar el Detector y el Visor.

> Requiere **Python 3.11.x** (no 3.12+). Si no lo tienes, descárgalo desde
> [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/)
> marcando *"Add Python to PATH"*.

## Uso

Doble clic en el acceso directo **Poly-X** del Escritorio, o en:

```bat
iniciar_polyx.bat
```

Abre el **Launcher** → selecciona el módulo.

O directamente desde terminal:

```bash
.venv\Scripts\python.exe -m polyx.launcher
.venv\Scripts\python.exe -m polyx.detector
.venv\Scripts\python.exe -m polyx.trainer
.venv\Scripts\python.exe -m polyx.etiquetador
.venv\Scripts\python.exe -m polyx.visor
```

## Actualizar

Para traer la última versión publicada en GitHub **sin reinstalar nada**, doble clic en:

```bat
actualizar.bat
```

Comprueba si hay un commit nuevo en `main`; si lo hay, descarga y reemplaza solo los
archivos del programa. **Conserva** tu entorno `.venv`, tus modelos `models\*.pt`, tus
`runs\` y cualquier dato local. No necesita tener Git instalado (descarga por HTTPS) —
solo conexión a internet.

### Aviso automático

No hace falta acordarse de ejecutarlo: al abrir el launcher se comprueba en segundo
plano si GitHub va por delante y, si es así, aparece un botón con el identificador de
la versión nueva. La comprobación no retrasa el arranque y se calla ante cualquier
fallo — sin internet o con GitHub caído simplemente no aparece el aviso.

### Si el equipo ya tenía Poly-X

`SETUP.bat` busca otras instalaciones al final del proceso. Si encuentra alguna,
muestra qué contiene —modelos, entrenamientos, detecciones, datasets, con su tamaño—
y pide confirmación antes de tocar nada.

El orden es **copiar → verificar → retirar**, nunca al revés:

- Se copian los datos que la instalación nueva no tenga. **No se sobrescribe ningún
  archivo existente.**
- Si algo falla al copiar, la carpeta vieja se conserva intacta y se avisa.
- Solo si todo salió bien, la carpeta antigua va a la **papelera**. No se borra: si
  falta algo, se restaura.

Las carpetas con `.git` se omiten siempre: un repositorio de desarrollo no es una
instalación vieja, y retirarlo se llevaría el trabajo sin versionar.

---

## Estructura del proyecto

```
polyx/
├── launcher.py          # Menú principal
├── core/                # Módulos compartidos (theme, yolo_wrap, metrics, report_html)
├── detector/            # Módulo 2: análisis en lote (9 páginas)
├── trainer/             # Módulo 3: entrenamiento YOLO (9 páginas)
├── etiquetador/         # Módulo 4: anotación interactiva
└── visor/               # Módulo 5: inspección + calibración μm/px
models/                  # Pesos .pt entrenados
runs/                    # Resultados de cada ejecución
data_microplastico/      # Dataset YOLO (images/ + labels/)
```

---

## Flujo completo de uso

```
Imágenes de microscopio (UV 254 nm, tinción Nile Red)
        ↓
  🏷 Etiquetador → anota PET/PP/LDPE en formato YOLO
        ↓
  🎯 Entrenador  → entrena modelo YOLO v8/v11
        ↓
  🔬 Detector    → analiza en lote + reporte HTML
        ↓
  📐 Visor       → inspección detallada + calibración μm/px
```

---

## Publicaciones

- **Pérez M, Parra S, Ferrada C, Bravo M, Pérez PA, Quiroz W (2024).** Development of a new methodology for the determination of PET microplastics in sediment, based on microwave-assisted acid digestion. *PLoS ONE* **19**(12): e0314520. https://doi.org/10.1371/journal.pone.0314520

- **Ferrada C, Pérez M, Parra S, Salas E, Sepúlveda F, Bravo MA, Quiroz W (2024).** Evaluation of microwave-assisted acid/oxidant digestion method for the detection of polyethylene microplastics in *Merluccius gayi* fish by Nile Red fluorescent staining and image analysis. *J. Chil. Chem. Soc.* **69**(1): 6082–6085. https://doi.org/10.4067/s0717-97072024000106082

---

## Manual de usuario

El archivo `Manual_PolyX.html` contiene la documentación completa con capturas de cada módulo, atajos de teclado, flujos de trabajo recomendados y referencias bibliográficas.

---

## Licencia y contacto

**Cristofher Ferrada**  
Laboratorio de Química Ambiental · Pontificia Universidad Católica de Valparaíso  
[cristofher.ferrada@pucv.cl](mailto:cristofher.ferrada@pucv.cl)
