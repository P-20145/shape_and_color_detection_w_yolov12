"""
validate.py - mAP@0.5, precision, recall y matriz de confusion (global y por clase).

Ejemplos:
    python validate.py                 # sobre el split de validacion
    python validate.py --split test    # sobre test (el que ningun ajuste ha tocado)

Salida:
    metrics/metrics_val.json (o metrics_test.json)
    metrics/figuras/<split>/confusion_matrix.png, PR_curve.png, etc.
"""
import argparse
import json
from pathlib import Path

from common import CLASSES, ensure_data_yaml

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "shapes3b"))
    ap.add_argument("--split", default="val", choices=["val", "test"], help="val = carpeta valid/")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    data_yaml = ensure_data_yaml(args.data)
    model = YOLO(args.weights)

    tag = "valid" if args.split == "val" else "test"
    fig_root = ROOT / "metrics" / "figuras"
    m = model.val(
        data=str(data_yaml), split=args.split, imgsz=args.imgsz, plots=True,
        project=str(fig_root), name=tag, exist_ok=True,   # matriz de confusion y curvas PR
    )

    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[m.names[int(c)]] = {
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
            "ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i]),
        }
    result = {
        "split": tag, "weights": str(args.weights),
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "per_class": per_class,
    }

    print(f"\n=== Resultados en {tag} ===")
    print(f"mAP@0.5    : {result['map50']:.3f}")
    print(f"mAP@0.5:95 : {result['map50_95']:.3f}")
    print(f"Precision  : {result['precision']:.3f}")
    print(f"Recall     : {result['recall']:.3f}")
    for name, v in per_class.items():
        print(f"  {name:10s} P={v['precision']:.3f}  R={v['recall']:.3f}  AP50={v['ap50']:.3f}")

    out = ROOT / "metrics" / f"metrics_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nGuardado: {out}")
    print(f"Matriz de confusion y curvas en: {fig_root / tag}")


if __name__ == "__main__":
    main()
