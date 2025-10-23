import csv
import math
import os
from typing import List, Tuple
import numpy as np


INPUT_CSV = os.path.join('data', 'data.csv')
OUTPUT_CSV = os.path.join('data', 'data_clean.csv')
DROP_Z = False  # set True if you want 2D-only features


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _to_float_list(values: List[str]) -> List[float]:
    return [float(v) for v in values]


def _read_rows(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    if any(('x' in c.lower() or 'label' in c.lower()) for c in header) or not _is_number(header[0]):
        data_rows = rows[1:]
    else:
        data_rows = rows
        header = []
    return header, data_rows


def _center_and_scale(flat_xyz: List[float]) -> List[float]:
    if len(flat_xyz) != 63:
        raise ValueError('Expected 63 features (21*3)')
    wx, wy, wz = flat_xyz[0], flat_xyz[1], flat_xyz[2]
    centered = []
    for i in range(0, 63, 3):
        x = flat_xyz[i] - wx
        y = flat_xyz[i + 1] - wy
        z = flat_xyz[i + 2] - wz
        centered.extend([x, y, z])
    max_dist = 0.0
    for i in range(0, 63, 3):
        dx, dy, dz = centered[i], centered[i + 1], centered[i + 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d > max_dist:
            max_dist = d
    if max_dist <= 1e-8:
        raise ValueError('Degenerate sample (zero scale)')
    for i in range(63):
        centered[i] /= max_dist
    if DROP_Z:
        xy = []
        for i in range(0, 63, 3):
            xy.extend([centered[i], centered[i + 1]])
        return xy
    return centered


def _default_header(drop_z: bool) -> List[str]:
    if drop_z:
        cols = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y']]
    else:
        cols = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
    cols.append('label')
    return cols


def main():
    if not os.path.exists(INPUT_CSV):
        print(f'Missing input CSV: {INPUT_CSV}')
        return

    header, data_rows = _read_rows(INPUT_CSV)
    if not data_rows:
        print(f'No data rows found in {INPUT_CSV}')
        return

    kept = 0
    skipped = 0
    cleaned: List[List[float]] = []

    for row in data_rows:
        if len(row) < 64:
            skipped += 1
            continue
        try:
            feats = _to_float_list(row[:63])
            label = int(float(row[63]))
            norm = _center_and_scale(feats)
            cleaned.append(norm + [label])
            kept += 1
        except Exception:
            skipped += 1

    out_header = _default_header(DROP_Z)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        writer.writerows(cleaned)

    total = kept + skipped
    print(f'Input rows: {total}')
    print(f'Kept: {kept}')
    print(f'Skipped: {skipped}')
    print(f'Wrote cleaned data to {OUTPUT_CSV}')

    # Also export NumPy arrays for training convenience
    if kept > 0:
        data = np.genfromtxt(OUTPUT_CSV, delimiter=',', skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        X = data[:, :-1].astype(np.float32)
        y = data[:, -1].astype(np.int64)
        np.save(os.path.join('data', 'X.npy'), X)
        np.save(os.path.join('data', 'y.npy'), y)
        print(f'Saved X to data/X.npy {X.shape}, y to data/y.npy {y.shape}')


if __name__ == '__main__':
    main()
