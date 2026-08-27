from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CLASS_ORDER = ["underdamped", "near_critical", "overdamped"]
FEATURE_COLUMNS = [
    "max_abs",
    "final_abs",
    "mean_abs",
    "std_response",
    "zero_crossings",
]


def load_paired_features(clean_path, noisy_path):
    clean = pd.read_csv(clean_path).set_index("sample_id").sort_index()
    noisy = pd.read_csv(noisy_path).set_index("sample_id").sort_index()
    if not clean.index.equals(noisy.index):
        raise ValueError("Clean and noisy feature files do not contain the same samples.")
    if not clean["label"].equals(noisy["label"]):
        raise ValueError("Clean and noisy feature labels are not aligned.")
    return clean, noisy


def paired_split(labels, test_size=0.2, seed=42):
    sample_ids = labels.index.to_numpy()
    train_ids, test_ids = train_test_split(
        sample_ids,
        test_size=test_size,
        random_state=seed,
        stratify=labels.to_numpy(),
    )
    return train_ids, test_ids


def fit_model(features, labels, sample_ids, seed=42):
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(features.loc[sample_ids, FEATURE_COLUMNS], labels.loc[sample_ids])
    return model


def evaluate(model, frame, sample_ids):
    actual = frame.loc[sample_ids, "label"]
    predicted = model.predict(frame.loc[sample_ids, FEATURE_COLUMNS])
    return actual, predicted


def save_confusion_matrix(actual, predicted, title, output_path):
    display = ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels=CLASS_ORDER,
        display_labels=["Underdamped", "Near critical", "Overdamped"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.figure_.savefig(output_path, dpi=180)
    plt.close(display.figure_)


def metric_row(scenario, train_dataset, test_dataset, actual, predicted, seed):
    return {
        "scenario": scenario,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro"),
        "test_samples": len(actual),
        "split_seed": seed,
    }


def run_evaluations(
    clean_path=DATA_DIR / "features.csv",
    noisy_path=DATA_DIR / "features_noisy.csv",
    results_dir=RESULTS_DIR,
    seed=42,
):
    clean, noisy = load_paired_features(clean_path, noisy_path)
    train_ids, test_ids = paired_split(clean["label"], seed=seed)

    clean_model = fit_model(clean, clean["label"], train_ids, seed=seed)
    noisy_model = fit_model(noisy, noisy["label"], train_ids, seed=seed)

    scenarios = [
        (
            "clean_holdout",
            "clean",
            "clean",
            *evaluate(clean_model, clean, test_ids),
            "Clean-trained model on clean holdout",
            "confusion_matrix.png",
        ),
        (
            "clean_to_noisy",
            "clean",
            "noisy",
            *evaluate(clean_model, noisy, test_ids),
            "Clean-trained model on paired noisy holdout",
            "confusion_matrix_transfer.png",
        ),
        (
            "noisy_holdout",
            "noisy",
            "noisy",
            *evaluate(noisy_model, noisy, test_ids),
            "Noisy-trained model on noisy holdout",
            "confusion_matrix_noisy.png",
        ),
    ]

    results_dir = Path(results_dir)
    rows = []
    for scenario, train_set, test_set, actual, predicted, title, filename in scenarios:
        rows.append(
            metric_row(scenario, train_set, test_set, actual, predicted, seed)
        )
        save_confusion_matrix(actual, predicted, title, results_dir / filename)

    metrics = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "evaluation_metrics.csv", index=False)
    return metrics


def main():
    metrics = run_evaluations()
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
