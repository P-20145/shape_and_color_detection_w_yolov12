"""
common.py - Funciones compartidas por train.py, validate.py, metrics_val.py y prepare_data.py.

Aqui vive la logica "pura" (sin YOLO ni camara) para poder probarla por separado:
  - ensure_data_yaml : genera el data.yaml con la ruta correcta en ESTE computador
  - iou              : interseccion sobre union entre dos cajas
  - match_detections : empareja predicciones con Ground Truth (base del Acc_val)
"""
from pathlib import Path

# Orden = id de clase en las etiquetas YOLO (0 = circle, 1 = square, 2 = triangle)
CLASSES = ["circle", "square", "triangle"]
SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_data_yaml(data_dir):
    """Crea (o regenera) data_dir/data.yaml con la ruta absoluta de este equipo.

    El data.yaml que sale de Colab trae rutas como /content/shapes3b, que no existen
    en el portatil. Por eso se reescribe cada vez que se corre un script.
    """
    data_dir = Path(data_dir).resolve()
    for split in SPLITS:
        if not (data_dir / split / "images").is_dir():
            raise FileNotFoundError(
                f"No existe {data_dir / split / 'images'}. "
                "Revisa el README, seccion 'Dataset': la carpeta debe quedar como data/shapes3b/{train,valid,test}/..."
            )
    yaml_path = data_dir / "data.yaml"
    yaml_path.write_text(
        f'path: "{data_dir.as_posix()}"\n'
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )
    return yaml_path


def xywhn_to_xyxy(box, img_w, img_h):
    """Formato YOLO (x_center, y_center, w, h normalizados) -> (x1, y1, x2, y2) en pixeles."""
    xc, yc, bw, bh = box
    return (
        (xc - bw / 2) * img_w,
        (yc - bh / 2) * img_h,
        (xc + bw / 2) * img_w,
        (yc + bh / 2) * img_h,
    )


def iou(a, b):
    """IoU entre dos cajas (x1, y1, x2, y2)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_detections(gts, preds, iou_thr=0.5):
    """Empareja predicciones con Ground Truth, uno a uno.

    gts   : lista de (clase, caja_xyxy)
    preds : lista de (clase, confianza, caja_xyxy)

    Una prediccion es CORRECTA si empareja un GT de su misma clase con IoU >= iou_thr.
    Cada GT solo puede ser cubierto por una prediccion (las de mayor confianza van primero),
    asi que una caja duplicada cuenta como falso positivo y no infla el resultado.

    Devuelve (indices_de_GT_cubiertos, numero_de_falsos_positivos).
    """
    used = set()
    false_pos = 0
    for cls, _conf, box in sorted(preds, key=lambda p: -p[1]):
        best_iou, best_i = 0.0, -1
        for i, (gt_cls, gt_box) in enumerate(gts):
            if i in used or gt_cls != cls:
                continue
            value = iou(box, gt_box)
            if value > best_iou:
                best_iou, best_i = value, i
        if best_i >= 0 and best_iou >= iou_thr:
            used.add(best_i)
        else:
            false_pos += 1
    return used, false_pos
