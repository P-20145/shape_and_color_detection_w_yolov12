"""
metrics_val.py - Calcula Accuracy_val, el numero que se muestra en el video (overlay fijo).

Definicion (enunciado): una deteccion es "correcta" si empareja un Ground Truth con
IoU >= 0.5 y clase correcta.

    Acc_val    = TP / GT_total            (de todas las figuras reales, cuantas se detectaron bien)
    Acc_estric = TP / (GT_total + FP)     (ademas penaliza detecciones sobrantes o equivocadas)

El emparejamiento es uno a uno: cada Ground Truth solo se puede usar una vez (ver common.py).
Se usa el mismo umbral de confianza que la demo en tiempo real (conf=0.25).

Ejemplos:
    python metrics_val.py                 # split valid -> metrics/acc_val.json (lo lee realtime.py)
    python metrics_val.py --split test    # split test  -> metrics/acc_test.json
"""
import argparse
import json
from pathlib import Path

from common import CLASSES, IMG_EXT, iou, match_detections, xywhn_to_xyxy  # noqa: F401

ROOT = Path(__file__).resolve().parent


def read_gt(label_file, img_w, img_h):
    """Lee las etiquetas YOLO de una imagen -> [(clase, caja_xyxy en pixeles)]."""
    gts = []
    if label_file.exists():
        for ln in label_file.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                gts.append((int(p[0]), xywhn_to_xyxy(list(map(float, p[1:5])), img_w, img_h)))
    return gts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "shapes3b"))
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--conf", type=float, default=0.25, help="umbral de confianza (igual que la demo)")
    ap.add_argument("--iou", type=float, default=0.5, help="umbral de IoU del enunciado")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", default=str(ROOT / "metrics"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    split_dir = Path(args.data) / args.split
    images = sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"No hay imagenes en {split_dir / 'images'}")

    total_gt = tp = fp = 0
    gt_by_class, tp_by_class = {}, {}
    for img in images:
        r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        h, w = r.orig_shape
        gts = read_gt(split_dir / "labels" / f"{img.stem}.txt", w, h)
        preds = list(zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()))

        matched, n_fp = match_detections(gts, preds, args.iou)
        total_gt += len(gts)
        tp += len(matched)
        fp += n_fp
        for i, (cls, _box) in enumerate(gts):
            gt_by_class[cls] = gt_by_class.get(cls, 0) + 1
            if i in matched:
                tp_by_class[cls] = tp_by_class.get(cls, 0) + 1

    acc_val = tp / total_gt if total_gt else 0.0
    acc_strict = tp / (total_gt + fp) if (total_gt + fp) else 0.0
    per_class = {
        model.names[c]: {"gt": n, "tp": tp_by_class.get(c, 0), "acc": round(tp_by_class.get(c, 0) / n, 4)}
        for c, n in sorted(gt_by_class.items())
    }

    result = {
        "acc_val": round(acc_val, 4),          # <- numero que lee realtime.py
        "acc_strict": round(acc_strict, 4),
        "split": args.split, "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
        "n_images": len(images), "gt": total_gt, "tp": tp, "fp": fp,
        "per_class": per_class,
        "formula": "acc_val = TP / GT_total ; acc_strict = TP / (GT_total + FP)",
    }
    print(f"Imagenes={len(images)}  GT={total_gt}  TP={tp}  FP={fp}")
    print(f"Acc_val        = {acc_val:.3f}  ({acc_val * 100:.1f}%)")
    print(f"Acc_estricta   = {acc_strict:.3f}  ({acc_strict * 100:.1f}%)")
    for name, v in per_class.items():
        print(f"  {name:10s} {v['tp']}/{v['gt']}  = {v['acc']:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ("acc_val.json" if args.split == "valid" else f"acc_{args.split}.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
