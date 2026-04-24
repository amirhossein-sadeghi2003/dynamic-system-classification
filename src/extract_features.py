import numpy as np
import pandas as pd


def count_zero_crossings(signal):
    count = 0
    for i in range(1, len(signal)):
        if signal[i - 1] * signal[i] < 0:
            count += 1
    return count


def extract_features(row):
    response = row[[col for col in row.index if col.startswith("x_")]].values.astype(float)

    features = {
        "max_abs": np.max(np.abs(response)),
        "final_abs": np.abs(response[-1]),
        "mean_abs": np.mean(np.abs(response)),
        "std_response": np.std(response),
        "zero_crossings": count_zero_crossings(response),
        "label": row["label"]
    }

    return features


def process_file(input_path, output_path):
    df = pd.read_csv(input_path)

    feature_rows = []
    for _, row in df.iterrows():
        feature_rows.append(extract_features(row))

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(output_path, index=False)

    print(f"Features saved to {output_path}")
    print(feature_df.head())


def main():
    process_file("data/simulation_data.csv", "data/features.csv")
    print()
    process_file("data/simulation_data_noisy.csv", "data/features_noisy.csv")


if __name__ == "__main__":
    main()
