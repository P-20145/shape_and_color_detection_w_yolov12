# Deteccion de formas geometricas con YOLOv12

Proyecto de Corte - 2° Corte · Electiva IV - Deep Computer Vision · Facultad de Ingenieria Mecatronica, Universidad Santo Tomas Bucaramanga.

**Autores:** `[COMPLETAR: nombres]`

Modelo **YOLOv12n** (Ultralytics) entrenado para detectar **circulo, cuadrado y triangulo**, con una demo en tiempo real por webcam que muestra bounding box, clase, confianza, color detectado y el Accuracy de validacion (`Acc_val`), todo al mismo tiempo.

---

## 1. Estructura del proyecto

```
yolo_formas/
├── data/                 dataset (data/shapes3b/{train,valid,test}) - ver seccion 4
├── models/               best.pt  (pesos del mejor modelo)
├── metrics/              acc_val.json, metrics_*.json
│   └── figuras/          matriz de confusion, curvas PR, resultados del entrenamiento
├── common.py             funciones compartidas (IoU, emparejamiento, data.yaml)
├── prepare_data.py       descarga, filtra y divide el dataset (opcional, ver seccion 4)
├── train.py              entrenamiento
├── validate.py           mAP@0.5, precision, recall, matriz de confusion
├── metrics_val.py        calcula Acc_val (el numero del overlay)
├── realtime.py           demo en tiempo real
├── requirements.txt
└── README.md
```

## 2. Instalacion

Python 3.9 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

La demo funciona en CPU. El entrenamiento se hizo en Google Colab con GPU Tesla T4.

## 3. Resumen de resultados

> Completar con los numeros finales de `metrics/metrics_valid.json`, `metrics/metrics_test.json` y `metrics/acc_val.json`.

| Metrica | Validacion | Test |
|---|---|---|
| mAP@0.5 | 0.955 | `[COMPLETAR]` |
| mAP@0.5:0.95 | 0.916 | `[COMPLETAR]` |
| Precision | 0.886 | `[COMPLETAR]` |
| Recall | 0.927 | `[COMPLETAR]` |
| **Acc_val** (TP / GT) | `[COMPLETAR]` | `[COMPLETAR]` |

Por clase (validacion, 170 imagenes, 231 instancias):

| Clase | Precision | Recall | AP@0.5 |
|---|---|---|---|
| circle | 0.877 | 0.957 | 0.972 |
| square | 0.856 | 0.867 | 0.932 |
| triangle | 0.926 | 0.959 | 0.962 |

La matriz de confusion y las curvas PR estan en `metrics/figuras/`.

## 4. Dataset

- **Origen:** [Shapes Classification](https://universe.roboflow.com/thesis-95twb/shapes-classification) (autor: *Thesis*, Roboflow Universe), licencia **CC BY 4.0**.
- **No fue generado por nosotros:** ya existia en el repositorio. El original tiene 3.738 imagenes y 12 clases (circle, square, triangle, rectangle, diamond, oval, cube, sphere, cone, cylinder, pyramid, heart).
- **Formato:** YOLO (un `.txt` por imagen con `class x_center y_center width height`, relativos).
- **Filtro estricto a 3 clases (circle, square, triangle):** se conservan solo las imagenes donde *todas* las etiquetas pertenecen a esas 3 clases. Si una imagen tenia ademas, por ejemplo, un rombo o un cubo, se descarta; de lo contrario esas figuras quedarian sin etiqueta y el modelo las aprenderia como fondo.
- **Re-division 70/20/10** (train/valid/test) con semilla 42, mezclando los tres splits originales.
- **Conteos finales:** `[COMPLETAR con la salida de la celda de conteo: imagenes e instancias por split]`

### Obtener el dataset (dos opciones)

**Opcion A (recomendada): usar exactamente el dataset con el que se entreno.**
Esto garantiza que `Acc_val` y las metricas se calculen sobre imagenes que el modelo *no* vio. En Colab, despues de preparar el dataset:

```python
!cd /content && zip -qr /content/drive/MyDrive/yolo_shapes/shapes3b.zip shapes3b
```

Descarga `shapes3b.zip` de Drive y extraelo dentro de `data/`, de modo que quede `data/shapes3b/train/images/...` (cuidado con que no quede `data/shapes3b/shapes3b/`). El `data.yaml` que trae tiene rutas de Colab; no importa, los scripts lo regeneran.

**Opcion B: regenerarlo con `prepare_data.py`.**

```bash
set ROBOFLOW_API_KEY=tu_clave        # Windows   (Mac/Linux: export ROBOFLOW_API_KEY=tu_clave)
python prepare_data.py --version 1   # la misma version de Roboflow que se uso al descargar
```

> **Cuidado:** el split puede no ser identico al de Colab. Si usan la opcion B, hay que **volver a entrenar** (seccion 5); validar un modelo viejo con un split nuevo mezcla imagenes que ya vio y infla las metricas.

## 5. Entrenamiento

```bash
python train.py                                  # yolo12n, imgsz=640, 100 epocas, early stopping (patience=20)
python train.py --model yolo12s.pt --name shapes3b_s   # variante small, para comparar
```

Parametros principales: `imgsz=640`, `epochs=100`, `batch=16`, `patience=20` (early stopping), `seed=42`. Modelo base `yolo12n.pt` (preentrenado en COCO, se descarga solo). Al terminar, el mejor modelo queda en **`models/best.pt`** y los graficos en `metrics/figuras/train/`.

En Colab se uso el mismo comando (o el equivalente con `model.train(...)`), guardando los resultados en Google Drive para no perderlos si la sesion se desconecta.

## 6. Validacion y metricas

```bash
python validate.py                  # mAP@0.5, precision, recall, matriz de confusion (valid)
python validate.py --split test     # lo mismo sobre test
python metrics_val.py               # Acc_val sobre valid -> metrics/acc_val.json
python metrics_val.py --split test  # Acc sobre test      -> metrics/acc_test.json
```

**Definicion de Acc_val** (la del enunciado): una deteccion es *correcta* si empareja un Ground Truth con **IoU ≥ 0.5** y **clase correcta**.

```
Acc_val    = TP / GT_total           (cuantas figuras reales se detectaron bien)
Acc_estric = TP / (GT_total + FP)    (ademas penaliza detecciones sobrantes o equivocadas)
```

- Umbral de confianza `conf = 0.25`, el mismo que usa la demo.
- Emparejamiento uno a uno: cada Ground Truth solo puede ser cubierto por una prediccion (las de mayor confianza primero); una caja duplicada cuenta como falso positivo.
- El overlay de la demo muestra `Acc_val`; `Acc_estric` queda en el JSON como referencia.

## 7. Demo en tiempo real

```bash
python realtime.py            # camara 0
python realtime.py --cam 1    # si abre otra camara
```

Muestra, en la misma ventana: bounding box, etiqueta con **clase + confianza + color detectado**, y el overlay fijo `Acc_val=XX%` leido de `metrics/acc_val.json` (no esta escrito a mano). Teclas: `q`/`ESC` salir, `s` guardar captura en `capturas/`.

El **color** se estima con la mediana HSV de la zona central de la caja (no lo aprende el modelo). Funciona mejor con figuras rellenas.

## 8. Reproducir las metricas (paso a paso)

1. Instalar dependencias (seccion 2).
2. Dejar el dataset en `data/shapes3b/` (seccion 4, opcion A) y `best.pt` en `models/`.
3. `python validate.py` → mAP@0.5, precision, recall, matriz de confusion.
4. `python metrics_val.py` → `Acc_val`.
5. `python realtime.py` → ver el overlay con el mismo `Acc_val`.

## 9. Experimentos realizados

| Experimento | Filtro de clases | Epocas | mAP@0.5 (valid) | AP@0.5 square |
|---|---|---|---|---|
| 1 | Conserva imagenes con otras figuras, borrando solo sus etiquetas | 50 (sin converger) | 0.759 | 0.612 |
| 2 (final) | **Estricto**: solo imagenes con 3 clases | hasta 100, patience=20 | 0.955 | 0.932 |

Nota: el split cambio entre experimentos (valid de 129 vs 170 imagenes), asi que la comparacion es informativa pero no exacta.

## 10. Limites y trabajo futuro

> Ajustar con lo que observen en la demo con camara real.

- El dataset mezcla imagenes de internet; el rendimiento con camara, iluminacion y fondos propios puede ser menor que en validacion.
- Solo 3 clases; figuras parecidas (rombo, rectangulo) no estan contempladas.
- El color se obtiene con una regla sobre HSV, sensible a la iluminacion.
- `[COMPLETAR: trabajo futuro, p. ej. fotos propias con webcam, mas clases, comparar yolo12n vs yolo12s]`

## 11. Creditos y cita del dataset

Dataset: *Shapes Classification*, Thesis, Roboflow Universe (CC BY 4.0). Modelo: YOLOv12 via [Ultralytics](https://github.com/ultralytics/ultralytics).

```bibtex
@misc{ shapes-classification_dataset,
  title = { Shapes Classification Dataset },
  type = { Open Source Dataset },
  author = { Thesis },
  howpublished = { \url{ https://universe.roboflow.com/thesis-95twb/shapes-classification } },
  url = { https://universe.roboflow.com/thesis-95twb/shapes-classification },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2025 },
  month = { dec },
  note = { visited on 2026-10-08 },
}
```
