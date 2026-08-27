from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generate_data import TIME_S


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CLASS_ORDER = ["underdamped", "near_critical", "overdamped"]


def response_columns(columns):
    return sorted(
        (column for column in columns if column.startswith("x_")),
        key=lambda column: int(column.split("_")[1]),
    )


def representative_row(frame, label):
    block = frame[frame["label"] == label].sort_values("damping_n_s_per_m")
    if block.empty:
        raise ValueError(f"No samples found for label: {label}")
    return block.iloc[len(block) // 2]


def plot_response(axis, row, columns, title):
    axis.plot(TIME_S, row[columns].to_numpy(dtype=float), linewidth=2)
    axis.axhline(0.0, linewidth=1, alpha=0.4)
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Displacement (m)")
    axis.grid(True, alpha=0.3)
    axis.text(
        0.96,
        0.92,
        f"c = {row['damping_n_s_per_m']:.3f} N s/m",
        transform=axis.transAxes,
        horizontalalignment="right",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def create_gallery(
    clean_path=DATA_DIR / "simulation_data.csv",
    noisy_path=DATA_DIR / "simulation_data_noisy.csv",
    output_path=RESULTS_DIR / "response_gallery.png",
):
    clean = pd.read_csv(clean_path)
    noisy = pd.read_csv(noisy_path)
    columns = response_columns(clean.columns)
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    figure.suptitle("Paired mass-spring-damper responses")

    for column_index, label in enumerate(CLASS_ORDER):
        display_label = label.replace("_", " ").title()
        clean_row = representative_row(clean, label)
        noisy_row = noisy[noisy["sample_id"] == clean_row["sample_id"]].iloc[0]
        plot_response(axes[0, column_index], clean_row, columns, f"Clean: {display_label}")
        plot_response(axes[1, column_index], noisy_row, columns, f"Noisy: {display_label}")

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main():
    output_path = create_gallery()
    print(f"Saved {output_path.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
