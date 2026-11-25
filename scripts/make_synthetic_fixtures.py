"""Generate SYNTHETIC fundus-like test images that mimic the EyePACS file layout.

Why: tests and CI must run without the real (licensed, ~35 GB) dataset. These images only have to
exercise the code paths (file naming, quality flags, duplicates, preprocessing, a tiny training
run). They carry no clinical meaning, and metrics computed on them must NEVER be reported as
results.

Layout produced (same as the Kaggle data)::

    <out>/trainLabels.csv            image,level
    <out>/train/<patient>_<left|right>.jpeg
    <out>/SYNTHETIC_DATA_README.txt  a marker saying this data is synthetic

Deliberate edge cases (their file stems are written to ``edge_cases.json``):

* ``duplicate``: one file copied byte-for-byte to a second patient (same MD5, same grade).
* ``unreadable``: a ``.jpeg`` that is not an image.
* ``dark``: a very dark (under-exposed) fundus.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Grade probabilities roughly matching EyePACS (25,810/2,443/5,292/873/708).
GRADE_PROBS = np.array([0.735, 0.070, 0.151, 0.025, 0.019])
# Simulated "camera" resolutions (width, height): real EyePACS images vary a lot.
CAMERAS: tuple[tuple[int, int], ...] = ((320, 240), (300, 300), (360, 270))


@dataclass
class FixtureSet:
    """Paths and edge-case names of a generated fixture set."""

    root: Path
    labels_csv: Path
    image_dir: Path
    n_images: int
    duplicate: tuple[str, str] = ("", "")
    unreadable: str = ""
    dark: str = ""
    grades: dict[str, int] = field(default_factory=dict)


def _draw_fundus(rng: np.random.Generator, size: tuple[int, int], grade: int) -> np.ndarray:
    """Draw a fundus-like BGR image: dark border, orange disc, optic disc, vessels, lesions."""
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    cx, cy = w // 2 + int(rng.integers(-8, 9)), h // 2 + int(rng.integers(-8, 9))
    radius = int(min(w, h) * rng.uniform(0.40, 0.47))

    # Background retina with a radial fall-off (vignetting, like real fundus photos).
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    inside = dist <= 1.0
    base = np.array([rng.uniform(20, 45), rng.uniform(60, 95), rng.uniform(150, 200)])  # BGR
    falloff = np.clip(1.0 - 0.45 * dist**2, 0.0, 1.0)[..., None]
    retina = (base[None, None, :] * falloff).clip(0, 255)
    img[inside] = retina[inside].astype(np.uint8)
    noise = rng.normal(0, 4, img.shape)
    img = np.where(inside[..., None], (img + noise).clip(0, 255), 0).astype(np.uint8)

    # Optic disc.
    side = 1 if rng.random() < 0.5 else -1
    od = (cx + side * int(radius * 0.45), cy + int(rng.integers(-10, 11)))
    cv2.circle(img, od, max(4, radius // 7), (150, 200, 240), -1)

    # Vessels: dark red curves radiating from the optic disc.
    for _ in range(6):
        angle = rng.uniform(0, 2 * np.pi)
        pts = []
        for t in np.linspace(0, 1, 8):
            r = t * radius * 0.95
            a = angle + 0.4 * np.sin(3 * t + rng.uniform(0, 1))
            pts.append((int(od[0] + r * np.cos(a)), int(od[1] + r * np.sin(a))))
        cv2.polylines(img, [np.array(pts, np.int32)], False, (20, 30, 110), 2)

    # Lesions scale with grade: microaneurysms/haemorrhages (dark red), exudates (yellow).
    n_red = [0, 3, 12, 25, 35][grade]
    n_yellow = [0, 0, 6, 12, 18][grade]
    for _ in range(n_red):
        p = (cx + int(rng.normal(0, radius * 0.4)), cy + int(rng.normal(0, radius * 0.4)))
        cv2.circle(img, p, int(rng.integers(1, 4)), (10, 10, 90), -1)
    for _ in range(n_yellow):
        p = (cx + int(rng.normal(0, radius * 0.35)), cy + int(rng.normal(0, radius * 0.35)))
        cv2.circle(img, p, int(rng.integers(2, 5)), (90, 220, 240), -1)
    if grade == 4:  # neovascularisation: tangle of thin vessels
        for _ in range(10):
            a, b = rng.integers(-radius // 3, radius // 3, size=(2, 2))
            cv2.line(img, (cx + a[0], cy + a[1]), (cx + b[0], cy + b[1]), (40, 40, 160), 1)

    img[~inside] = 0
    return img


def generate_fixtures(out_dir: Path | str, n_patients: int = 30, seed: int = 0) -> FixtureSet:
    """Create the synthetic dataset. Deterministic for a given ``seed``."""
    if n_patients < 7:
        raise ValueError("Need at least 7 patients to place every edge case and both classes.")
    root = Path(out_dir)
    if root.exists():
        shutil.rmtree(root)
    image_dir = root / "train"
    image_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)

    patient_ids = sorted(int(x) for x in rng.choice(np.arange(10, 10_000), n_patients, replace=False))
    rows: list[tuple[str, int]] = []
    for i, pid in enumerate(patient_ids):
        # Guarantee both classes: first patients are forced referable / non-referable.
        if i == 0:
            patient_grade = 3
        elif i == 1:
            patient_grade = 2
        elif i == 2:
            patient_grade = 0
        else:
            patient_grade = int(rng.choice(5, p=GRADE_PROBS))
        camera = CAMERAS[int(rng.integers(len(CAMERAS)))]
        for eye in ("left", "right"):
            # Eyes of one patient are correlated but not identical.
            grade = patient_grade if rng.random() < 0.8 else max(0, patient_grade - 1)
            stem = f"{pid}_{eye}"
            img = _draw_fundus(rng, camera, grade)
            cv2.imwrite(str(image_dir / f"{stem}.jpeg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append((stem, grade))

    grades = dict(rows)
    stems = [s for s, _ in rows]

    # Edge case 1: byte-identical file shared by two different patients.
    src_stem, dst_stem = stems[6], stems[9]  # different patients by construction (2 eyes each)
    shutil.copyfile(image_dir / f"{src_stem}.jpeg", image_dir / f"{dst_stem}.jpeg")
    grades[dst_stem] = grades[src_stem]

    # Edge case 2: unreadable file with an image extension.
    unreadable = stems[-1]
    (image_dir / f"{unreadable}.jpeg").write_bytes(b"SYNTHETIC-NOT-AN-IMAGE\x00\x01\x02" * 8)

    # Edge case 3: very dark image.
    dark = stems[-3]
    dark_img = cv2.imread(str(image_dir / f"{dark}.jpeg"))
    cv2.imwrite(str(image_dir / f"{dark}.jpeg"), (dark_img * 0.08).astype(np.uint8))

    labels_csv = root / "trainLabels.csv"
    pd.DataFrame({"image": list(grades), "level": list(grades.values())}).to_csv(labels_csv, index=False)

    fs = FixtureSet(
        root=root,
        labels_csv=labels_csv,
        image_dir=image_dir,
        n_images=len(grades),
        duplicate=(src_stem, dst_stem),
        unreadable=unreadable,
        dark=dark,
        grades=grades,
    )
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(fs).items()}
    (root / "edge_cases.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (root / "SYNTHETIC_DATA_README.txt").write_text(
        "SYNTHETIC DATA generated by scripts/make_synthetic_fixtures.py for tests only.\n"
        "These are not retinal photographs. Never report metrics computed on them.\n",
        encoding="utf-8",
    )
    return fs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="tests/fixtures/synthetic", help="Output directory.")
    parser.add_argument("--n-patients", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    fs = generate_fixtures(args.out, n_patients=args.n_patients, seed=args.seed)
    print(f"Wrote {fs.n_images} SYNTHETIC images to {fs.image_dir} (labels: {fs.labels_csv}).")
    print(f"Edge cases -> duplicate: {fs.duplicate}, unreadable: {fs.unreadable}, dark: {fs.dark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
