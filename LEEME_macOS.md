# Poly-X en macOS

Suite de detección de microplásticos por fluorescencia Nile Red.
Cristofher Ferrada · Doctorado en Ciencias mención Química · PUCV

---

## Instalación (una sola vez)

1. Descarga el proyecto y descomprímelo donde quieras (por ejemplo, en Documentos).
2. **Clic derecho** sobre `Lanzar_macOS.command` → **Abrir** → **Abrir**.
3. Espera 10-15 minutos. Se instala todo solo.

A partir de ahí, doble clic normal en `Lanzar_macOS.command` cada vez que quieras usarlo.

### ⚠️ «No se puede abrir porque es de un desarrollador no identificado»

Es lo esperable y **no significa que esté roto**. macOS bloquea por defecto
cualquier script descargado de internet que no venga firmado por un
desarrollador registrado en Apple.

**Solución:** clic **derecho** sobre el archivo → **Abrir** → confirmar **Abrir**
en el diálogo. Solo hace falta la primera vez; después el doble clic funciona.

Si aun así se resiste, en Terminal:

```bash
xattr -d com.apple.quarantine Lanzar_macOS.command
```

### Requisito previo

**Python 3.9 o superior.** Compruébalo en Terminal con `python3 --version`.
Si no lo tienes, descárgalo de [python.org/downloads](https://www.python.org/downloads/).

---

## Rendimiento: qué esperar

Depende del procesador del Mac, y la diferencia es grande.

| Mac | Aceleración | Detección |
|---|---|---|
| **Apple Silicon** (M1, M2, M3, M4) | GPU integrada vía MPS | Rápida |
| **Intel** | Solo CPU | Lenta: cuenta ~1 min por foto en lotes con troceo |

El instalador detecta cuál tienes y configura el dispositivo solo. No hay que
tocar nada, pero conviene saberlo antes de lanzar un lote de 500 fotos en un
Mac Intel.

> **Nota para Mac Intel:** PyTorch dejó de publicar versiones para procesadores
> Intel después de la 2.2.2, así que el instalador fija esa versión. Es la
> última que existe para esa arquitectura y funciona correctamente.

---

## Actualizar

Doble clic en `actualizar_macOS.command`. Descarga la versión nueva de GitHub y
sobrescribe solo los archivos del programa: **tus modelos, resultados y datos no
se tocan**.

También aparece un botón «Actualizar» dentro de Poly-X cuando hay versión nueva.

---

## Poly-X.app (opcional)

`construir_app_macOS.command` empaqueta la aplicación en un `Poly-X.app`
arrastrable a Aplicaciones.

**No hace falta para usar Poly-X**, y tiene inconvenientes reales:

- Tarda 15-40 minutos y ocupa 2-5 GB (PyTorch va dentro del bundle).
- Sin firma de Apple (USD 99/año) Gatekeeper lo bloquea igual en otro Mac, así
  que **no evita** el aviso de «desarrollador no identificado».
- Empaquetar PyTorch con PyInstaller es frágil.

La vía recomendada es `Lanzar_macOS.command`.

---

## Si algo falla

| Síntoma | Qué pasa |
|---|---|
| `bad interpreter: /bin/bash^M` | El archivo llegó con finales de línea de Windows. Descárgalo otra vez desde GitHub, no lo copies por WhatsApp/correo. |
| Doble clic y no pasa nada | Falta el permiso de ejecución: `chmod +x Lanzar_macOS.command` en Terminal. |
| «Python 3.9+ no encontrado» | Instálalo desde python.org/downloads. |
| Falla al instalar PyTorch | Suele ser la conexión. Vuelve a ejecutar el lanzador: retoma desde donde iba. |
| La detección no encuentra nada | Baja la confianza a 0.10 en Parámetros. |

---

## Qué se ha probado y qué no

Honestidad sobre el estado de esta versión:

- **Verificado en Windows:** toda la lógica compartida — la app, el informe, la
  medición de partículas, los 49 tests automáticos.
- **Escrito para macOS pero pendiente de probar en un Mac:** los lanzadores, el
  actualizador, la detección de MPS y el empaquetado `.app`.

Si encuentras un fallo, anota el mensaje exacto de la ventana de Terminal: es lo
que permite arreglarlo.
