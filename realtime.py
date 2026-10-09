"""
realtime.py - Demo en tiempo real (webcam) con YOLOv12.

Muestra AL MISMO TIEMPO en la ventana de video:
  - bounding box de cada figura
  - etiqueta: clase + confianza + color detectado
  - overlay fijo con el Accuracy de validacion (Acc_val=XX%), leido de metrics/acc_val.json

El detector es YOLO. OpenCV solo se usa para abrir la camara, dibujar y estimar el color.

Uso:
    python realtime.py
    python realtime.py --cam 1        # si abre otra camara
Teclas:  q o ESC = salir   |   s = guardar captura en capturas/
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent

# Colores del recuadro por clase (BGR)
CLASS_COLORS = {0: (0, 200, 0), 1: (255, 120, 0), 2: (0, 140, 255)}


def color_name(crop_bgr):
    """Color dominante de un recorte: mediana en HSV (el matiz H va de 0 a 179 en OpenCV)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = (float(np.median(hsv[..., i])) for i in range(3))
    if v < 50:
        return "negro"
    if s < 40:
        return "blanco" if v > 170 else "gris"
    if h < 10 or h >= 170:
        return "rojo"
    if h < 22:
        return "naranja"
    if h < 35:
        return "amarillo"
    if h < 85:
        return "verde"
    if h < 130:
        return "azul"
    if h < 160:
        return "morado"
    return "rosa"


def open_camera(index):
    """Abre la camara; en Windows prueba primero DirectShow (abre mas rapido)."""
    cap = None
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def draw_detection(frame, box, cls_id, label):
    """Dibuja el recuadro y la etiqueta (fondo del mismo color que el recuadro)."""
    x1, y1, x2, y2 = box
    color = CLASS_COLORS.get(cls_id, (0, 255, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = y1 - th - 8 if y1 - th - 8 >= 0 else y1  # si no cabe arriba, va dentro del recuadro
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + 8), color, -1)
    cv2.putText(frame, label, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--acc-json", default=str(ROOT / "metrics" / "acc_val.json"))
    ap.add_argument("--cam", type=int, default=0, help="indice de la camara")
    ap.add_argument("--conf", type=float, default=None, help="por defecto el mismo conf usado al calcular Acc_val")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    # Acc_val se LEE del archivo (calculado en validacion), nunca se escribe a mano
    acc_path = Path(args.acc_json)
    if not acc_path.exists():
        sys.exit(f"No existe {acc_path}. Corre primero: python metrics_val.py")
    acc_info = json.loads(acc_path.read_text(encoding="utf-8"))
    acc_val = acc_info["acc_val"]
    conf = args.conf if args.conf is not None else acc_info.get("conf", 0.25)

    from ultralytics import YOLO

    model = YOLO(args.weights)

    cap = open_camera(args.cam)
    if not cap.isOpened():
        sys.exit("No se pudo abrir la camara. Prueba con --cam 1 y cierra otras apps que la usen (Zoom, Teams...).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps, prev, shots = 0.0, time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        # Inferencia YOLO sobre el frame
        result = model.predict(frame, conf=conf, imgsz=args.imgsz, verbose=False)[0]

        for b in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            cls_id, score = int(b.cls), float(b.conf)

            # Color: se mide solo en la parte central de la caja (evita el borde y el fondo)
            bw, bh = x2 - x1, y2 - y1
            crop = frame[y1 + bh // 4: y2 - bh // 4, x1 + bw // 4: x2 - bw // 4]
            color = color_name(crop) if crop.size else "?"

            draw_detection(frame, (x1, y1, x2, y2), cls_id, f"{model.names[cls_id]} {score:.2f} | {color}")

        # Overlay fijo: Accuracy de validacion + FPS (promedio suavizado)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6)) if fps else 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.rectangle(frame, (10, 10), (270, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Acc_val={acc_val * 100:.0f}%", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.0f}  conf>={conf:.2f}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("YOLOv12 - Formas geometricas (q = salir, s = captura)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            (ROOT / "capturas").mkdir(exist_ok=True)
            shots += 1
            cv2.imwrite(str(ROOT / "capturas" / f"captura_{shots}.png"), frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
