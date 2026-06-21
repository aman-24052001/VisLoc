"""
Battery model validation.

Reproduces Table III of Bauersfeld & Scaramuzza (2022) - six real
commercial drones, comparing this project's reimplementation against
both the paper's own "pen-and-paper" reproduction and the manufacturers'
published specifications. This is the same validation pattern used
throughout VisLoc: don't trust a model because the math looks right,
check it against an external ground truth.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visloc3d.battery import MulticopterSpec, estimate_range_and_endurance

# Table III, Bauersfeld & Scaramuzza 2022. Propeller column is radius in
# meters (confirmed by reproducing the paper's own worked Mavic 3 example:
# using the table value directly as radius gives v_i,h=4.50 m/s against
# their stated 4.51 m/s - matches almost exactly).
DRONES = [
    MulticopterSpec("DJI Mavic 2", 0.91, 4, 0.110, 4, 1, 3.9, 0.0200),
    MulticopterSpec("DJI Mavic 3", 0.90, 4, 0.119, 4, 1, 5.0, 0.0215),
    MulticopterSpec("DJI Matrice 200", 6.14, 4, 0.216, 6, 2, 15.3, 0.1700),
    MulticopterSpec("DJI Matrice 600 Pro", 15.5, 6, 0.267, 6, 6, 34.2, 0.1760),
    MulticopterSpec("Parrot Anafi AI", 0.90, 4, 0.057, 4, 1, 6.8, 0.0400),
    MulticopterSpec("Skydio 2", 0.78, 4, 0.085, 3, 1, 4.3, 0.0268),
]

# Reference values transcribed directly from the paper's Table III.
MFR_ENDURANCE_MIN = {"DJI Mavic 2": 31, "DJI Mavic 3": 46, "DJI Matrice 200": 24,
                      "DJI Matrice 600 Pro": 18, "Parrot Anafi AI": 32, "Skydio 2": 23}
PAPER_OURS_MIN = {"DJI Mavic 2": 33, "DJI Mavic 3": 48, "DJI Matrice 200": 23,
                  "DJI Matrice 600 Pro": 18, "Parrot Anafi AI": 31, "Skydio 2": 26}


def run_validation():
    rows = []
    for d in DRONES:
        est = estimate_range_and_endurance(d)
        ours_min = est.endurance_s / 60
        mfr = MFR_ENDURANCE_MIN[d.name]
        paper = PAPER_OURS_MIN[d.name]
        rows.append((d.name, ours_min, paper, mfr, 100 * (ours_min - mfr) / mfr))
    return rows


def plot_comparison(rows, out_path):
    names = [r[0] for r in rows]
    ours = [r[1] for r in rows]
    paper = [r[2] for r in rows]
    mfr = [r[3] for r in rows]

    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, mfr, w, label="Manufacturer spec", color="#888")
    ax.bar(x, paper, w, label="Paper's reproduction", color="#5b8def")
    ax.bar(x + w, ours, w, label="This project's reproduction", color="#f5b942")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Endurance (minutes)")
    ax.set_title("Battery/Range Model Validation - 6 Real Drones")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    rows = run_validation()
    print(f"{'Drone':<22}{'Ours':>8}{'Paper':>8}{'MfrSpec':>9}{'OursVsMfr':>12}")
    for name, ours, paper, mfr, err in rows:
        print(f"{name:<22}{ours:>7.1f}m{paper:>7}m{mfr:>8}m{err:>+11.1f}%")
    mean_abs_err = np.mean([abs(r[4]) for r in rows])
    print(f"\nMean |error| vs manufacturer spec: {mean_abs_err:.1f}%")
    plot_comparison(rows, "assets3d/battery_validation.png")
    print("Chart saved to assets3d/battery_validation.png")
