# Poly-X — Suite de detección de microplásticos

Plataforma de detección y clasificación de microplásticos (PET, PP, LDPE) por fluorescencia Nile Red bajo luz UV (254 nm) con modelos YOLO v8/v11.

> **Estado:** reconstrucción en curso (mayo 2026). Ver [`Manual_PolyX.html`](Manual_PolyX.html) para la especificación completa.

## Módulos

| Módulo | Descripción |
|---|---|
| 🏠 Launcher | Menú principal — lanza cada módulo en ventana independiente |
| 🔬 Detector | Procesamiento en lote con YOLO + reporte HTML paper-quality |
| 🎯 Entrenador | Entrenamiento YOLO v8/v11 con curvas en vivo |
| 🏷 Etiquetador | Anotación YOLO con pre-anotación automática |
| 📐 Visor | Inspección de imágenes con calibración interactiva μm/píxel |

## Requisitos

- Windows 10/11
- Python **3.11.x** (no 3.12+)
- 8 GB RAM mínimo · GPU NVIDIA opcional (recomendada)

## Instalación

```bat
SETUP.bat
```

Crea el entorno `.venv`, detecta GPU NVIDIA y descarga PyTorch (CUDA o CPU).

## Uso

```bat
iniciar_polyx.bat
```

## Autor

Cristofher Ferrada · Doctorado en Ciencias mención Química · 2026

## Referencias

- Pérez M, Parra S, Ferrada C, et al. (2024). *PLoS ONE* 19(12): e0314520. https://doi.org/10.1371/journal.pone.0314520
- Ferrada C, Pérez M, Parra S, et al. (2024). *J. Chil. Chem. Soc.* 69(1): 6082.
