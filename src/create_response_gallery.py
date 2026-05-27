from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

CLEAN_DATA = ROOT / "data" / "simulation_data.csv"
NOISY_DATA = ROOT / "data" / "simulation_data_noisy.csv"

OUTPUT = RESULTS_DIR / "response_gallery.png"

CLASSES = ["underdamped", "critical", "overdamped"]


def response_columns(df):
    return [col for col in df.columns if col.startswith("x_")]


def pick_representative(df, label):
    block = df[df["label"] == label].copy()

    if block.empty:
        raise ValueError(f"No samples found for label: {label}")

    if "c_value" in block.columns:
        block = block.sort_values("c_value")

    middle_index = len(block) // 2
    return block.iloc[middle_index]


def plot_response(ax, row, x_cols, title):
    y = row[x_cols].to_numpy(dtype=float)
    t = np.linspace(0, 1, len(y))

    ax.plot(t, y, linewidth=2)
    ax.axhline(0, linewidth=1, alpha=0.4)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("normalized time")
    ax.set_ylabel("displacement")
    ax.grid(True, alpha=0.3)

    if "c_value" in row:
        ax.text(
            0.03,
            0.92,
            f"c = {row['c_value']:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    clean_df = pd.read_csv(CLEAN_DATA)
    noisy_df = pd.read_csv(NOISY_DATA)

    x_cols = response_columns(clean_df)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    fig.suptitle(
        "Mass-Spring-Damper Response Gallery: Clean vs Noisy Signals",
        fontsize=16,
    )

    for col_index, label in enumerate(CLASSES):
        clean_row = pick_representative(clean_df, label)
        noisy_row = pick_representative(noisy_df, label)

        plot_response(
            axes[0, col_index],
            clean_row,
            x_cols,
            f"Clean {label}",
        )

        plot_response(
            axes[1, col_index],
            noisy_row,
            x_cols,
            f"Noisy {label}",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT, dpi=170)
    plt.close(fig)

    print("Saved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
