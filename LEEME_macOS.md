# Poly-X en macOS

Suite de detección de microplásticos por fluorescencia Nile Red.
Cristofher Ferrada · Dr (c) en Ciencias mención Química · PUCV

---

## Instalación (una sola vez)

1. Descarga el proyecto y descomprímelo donde quieras (por ejemplo, en Documentos).
2. **Clic derecho** sobre `Lanzar_macOS.command` → **Abrir** → **Abrir**.
3. Espera 10-15 minutos. Se instala todo solo.

A partir de ahí, doble clic normal en `Lanzar_macOS.command` cada vez que quieras usarlo.

Al terminar la instalación te ofrece dejar un **`Poly-X.app` en el Escritorio**:
acéptalo y usa ese para el día a día, que abre sin ventana de Terminal.

---

## Abrir sin la ventana negra de Terminal

`Lanzar_macOS.command` deja **siempre** una ventana de Terminal abierta detrás
del programa. No es un fallo: un `.command` es, por definición, un script que
Finder le entrega a la Terminal, y esa ventana vive mientras viva el programa.
Si la cierras, cierras Poly-X.

Para abrirlo sin ella, doble clic en **`crear_icono_macOS.command`**. Crea un
`Poly-X.app` —Escritorio, Aplicaciones o junto al programa, tú eliges— y a
partir de ahí abres desde ese icono. Es el equivalente de `Poly-X.vbs` en
Windows.

- Tarda **un segundo** y pesa unos KB: no empaqueta nada, solo llama al
  entorno que ya está instalado. Nada que ver con `construir_app_macOS.command`.
- Al crearse **en tu propio Mac** no lleva la marca de cuarentena, así que
  Gatekeeper no protesta: doble clic normal desde la primera vez.
- Sin consola no hay dónde leer los errores, así que todo lo que el programa
  escriba queda en `~/Library/Logs/Poly-X.log`.
- La ruta del programa va escrita dentro del `.app`. **Si mueves la carpeta de
  Poly-X, vuelve a ejecutar `crear_icono_macOS.command`** desde donde quedó.

Conserva `Lanzar_macOS.command`: es el que hay que abrir cuando algo falla,
porque enseña el error en pantalla en vez de esconderlo en el registro.

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

## Empaquetar con PyInstaller (opcional, y casi nunca necesario)

`construir_app_macOS.command` mete PyTorch, ultralytics y el intérprete entero
dentro de un `.app` autocontenido, para llevarlo a un Mac que no tenga nada
instalado.

> No lo confundas con `crear_icono_macOS.command`, que es lo que quieres si
> solo buscas abrir sin Terminal: aquel tarda un segundo, este entre 15 y 40
> minutos.

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
| El `Poly-X.app` rebota en el Dock y se cierra | Abre `~/Library/Logs/Poly-X.log`: ahí está el error que la Terminal habría mostrado. |
| El `.app` avisa de que no encuentra la instalación | Moviste la carpeta del programa. Vuelve a ejecutar `crear_icono_macOS.command` desde su ubicación nueva. |
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
