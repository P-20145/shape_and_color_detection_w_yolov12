"""
train.py - Entrena YOLOv12 (yolo12n o yolo12s) con el dataset de formas geometricas.

Ejemplos:
    python train.py                                   # yolo12n, 100 epocas, early stopping
    python train.py --model yolo12s.pt --name shapes3b_s
    python train.py --epochs 3 --name prueba          # prueba rapida

Al terminar copia el mejor modelo a models/best.pt y los graficos a metrics/figuras/train/.
"""
import argparse
import shutil
from pathlib import Path

from common import ensure_data_yaml

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "shapes3b"), help="carpeta con train/valid/test")
    ap.add_argument("--model", default="yolo12n.pt", help="yolo12n.pt o yolo12s.pt (se descarga solo)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="bajar a 8 si hay error de memoria")
    ap.add_argument("--patience", type=int, default=20, help="early stopping: epocas sin mejora antes de parar")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="0 para GPU, cpu para CPU (por defecto: automatico)")
    ap.add_argument("--project", default=str(ROOT / "runs"), help="carpeta de salida (en Colab: una ruta de Drive)")
    ap.add_argument("--name", default="shapes3b_n")
    args = ap.parse_args()

    from ultralytics import YOLO  # import local para que --help sea rapido

    data_yaml = ensure_data_yaml(args.data)
    model = YOLO(args.model)

    kwargs = dict(
        data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        patience=args.patience, workers=args.workers, seed=args.seed,
        project=str(Path(args.project).resolve()), name=args.name,
    )
    if args.device is not None:
        kwargs["device"] = args.device
    model.train(**kwargs)

    # Guardar los pesos finales (best) en models/ para la entrega
    save_dir = Path(model.trainer.save_dir)
    best = Path(model.trainer.best) if getattr(model.trainer, "best", None) else save_dir / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / "best.pt")
    print(f"\nMejor modelo copiado a models/best.pt (origen: {best})")

    # Copiar graficos de entrenamiento para el informe / presentacion
    fig_dir = ROOT / "metrics" / "figuras" / "train"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("results.*", "confusion_matrix*.png", "*curve.png", "labels*.jpg"):
        for f in save_dir.glob(pattern):
            shutil.copy(f, fig_dir / f.name)
    print(f"Graficos copiados a {fig_dir}")


if __name__ == "__main__":
    main()
