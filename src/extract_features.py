from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def response_columns(columns):
    return sorted(
        (column for column in columns if column.startswith("x_")),
        key=lambda column: int(column.split("_")[1]),
    )


def count_zero_crossings(signal):
    return int(np.count_nonzero(signal[:-1] * signal[1:] < 0.0))


def extract_features(row, columns):
    response = row[columns].to_numpy(dtype=float)
    return {
        "sample_id": row["sample_id"],
        "label": row["label"],
        "max_abs": np.max(np.abs(response)),
        "final_abs": np.abs(response[-1]),
        "mean_abs": np.mean(np.abs(response)),
        "std_response": np.std(response),
        "zero_crossings": count_zero_crossings(response),
    }


def process_file(input_path, output_path):
    frame = pd.read_csv(input_path)
    columns = response_columns(frame.columns)
    feature_frame = pd.DataFrame(
        extract_features(row, columns) for _, row in frame.iterrows()
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(output_path, index=False)
    return feature_frame


def main():
    clean = process_file(
        DATA_DIR / "simulation_data.csv", DATA_DIR / "features.csv"
    )
    noisy = process_file(
        DATA_DIR / "simulation_data_noisy.csv", DATA_DIR / "features_noisy.csv"
    )
    print(f"Extracted features for {len(clean)} clean and {len(noisy)} noisy responses.")


if __name__ == "__main__":
    main()
