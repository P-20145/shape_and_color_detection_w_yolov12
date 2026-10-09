"""
prepare_data.py - Descarga el dataset de Roboflow Universe, lo filtra a 3 clases y lo
re-divide en train/valid/test (70/20/10).

Dataset original: https://universe.roboflow.com/thesis-95twb/shapes-classification
(licencia CC BY 4.0, ya existia en el repositorio: NO fue generado por nosotros).

Filtro ESTRICTO: solo se conservan las imagenes donde TODAS las etiquetas son circle,
square o triangle. Si una imagen tiene ademas un rombo o un cubo, se descarta completa;
de lo contrario esas figuras quedarian sin etiqueta y el modelo las aprenderia como fondo.

ATENCION: el split depende del orden de lectura de los archivos. Si ya entrenaron en Colab,
NO regeneren el split aqui y validen con la carpeta que usaron en Colab; si lo hacen, el
modelo ya vio parte de las imagenes "nuevas" de validacion y el Acc_val saldria inflado.
Si regeneran el dataset aqui, vuelvan a entrenar con train.py.

Uso:
    set ROBOFLOW_API_KEY=tu_clave        (Windows)   |   export ROBOFLOW_API_KEY=tu_clave
    python prepare_data.py --version 1
"""
import argparse
import collections
import os
import random
import shutil
from pathlib import Path

import yaml

from common import CLASSES, IMG_EXT, SPLITS, ensure_data_yaml

WORKSPACE = "thesis-95twb"
PROJECT = "shapes-classification"


def download_raw(version, dest, api_key):
    """Descarga el dataset original en formato YOLO (TXT por imagen)."""
    from roboflow import Roboflow  # import local: solo se necesita para descargar

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(WORKSPACE).project(PROJECT).version(version).download("yolov8", location=str(dest))
    return Path(ds.location)


def collect_clean_samples(raw_root, keep):
    """Reune los 3 splits originales y deja solo imagenes con todas sus etiquetas en `keep`.

    Devuelve lista de (ruta_imagen, id_unico, filas_de_etiqueta_con_ids_remapeados).
    """
    raw_root = Path(raw_root)
    names = yaml.safe_load((raw_root / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    missing = [c for c in keep if c not in names]
    if missing:
        raise ValueError(f"Clases {missing} no estan en el dataset. Clases disponibles: {names}")

    # id original -> id nuevo (0, 1, 2)
    id_map = {names.index(c): i for i, c in enumerate(keep)}

    samples, seen = [], set()
    for split in SPLITS:
        img_dir, lbl_dir = raw_root / split / "images", raw_root / split / "labels"
        if not lbl_dir.is_dir():
            continue
        # Indice stem -> ruta (evita problemas de glob con nombres raros)
        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        for lf in sorted(lbl_dir.glob("*.txt")):
            rows = [ln.split() for ln in lf.read_text().splitlines() if ln.strip()]
            if not rows or lf.stem not in images:
                continue
            if all(int(r[0]) in id_map for r in rows):
                uid = lf.stem if lf.stem not in seen else f"{split}_{lf.stem}"
                seen.add(uid)
                new_rows = [" ".join([str(id_map[int(r[0])])] + r[1:]) for r in rows]
                samples.append((images[lf.stem], uid, new_rows))
    return samples


def split_samples(samples, ratios=(0.7, 0.2, 0.1), seed=42):
    """Mezcla con semilla fija y corta en train/valid/test."""
    ordered = sorted(samples, key=lambda s: s[1])  # orden base determinista
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    a, b = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)
    return {"train": ordered[:a], "valid": ordered[a:b], "test": ordered[b:]}


def write_dataset(splits, out_dir):
    """Copia imagenes y escribe las etiquetas ya remapeadas."""
    out_dir = Path(out_dir)
    for split, items in splits.items():
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img, uid, rows in items:
            shutil.copy(img, out_dir / split / "images" / f"{uid}{img.suffix}")
            (out_dir / split / "labels" / f"{uid}.txt").write_text("\n".join(rows), encoding="utf-8")


def print_counts(out_dir):
    """Imagenes e instancias por split y por clase (copiar al README)."""
    for split in SPLITS:
        counts = collections.Counter()
        files = list((Path(out_dir) / split / "labels").glob("*.txt"))
        for lf in files:
            for ln in lf.read_text().splitlines():
                if ln.strip():
                    counts[CLASSES[int(ln.split()[0])]] += 1
        print(f"{split:6s} imagenes={len(files):4d}  instancias={dict(counts)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", type=int, default=1, help="version del dataset en Roboflow (usar la misma que en Colab)")
    ap.add_argument("--raw", default="data/_raw_download", help="carpeta de descarga original")
    ap.add_argument("--out", default="data/shapes3b", help="carpeta del dataset final")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"), help="o variable ROBOFLOW_API_KEY")
    args = ap.parse_args()

    raw = Path(args.raw)
    if not (raw / "data.yaml").exists():
        if not args.api_key:
            raise SystemExit("Falta la API key: define la variable de entorno ROBOFLOW_API_KEY (no la escribas en el codigo).")
        raw = download_raw(args.version, raw, args.api_key)

    samples = collect_clean_samples(raw, CLASSES)
    print(f"Imagenes limpias (solo {CLASSES}): {len(samples)}")
    if len(samples) < 300:
        print("AVISO: hay menos de 300 imagenes; el enunciado exige >= 300.")

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    write_dataset(split_samples(samples, seed=args.seed), out)
    ensure_data_yaml(out)
    print_counts(out)
    print(f"Dataset listo en {out.resolve()}")


if __name__ == "__main__":
    main()
