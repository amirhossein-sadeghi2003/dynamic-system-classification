import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


def run_experiment(input_path, output_image, title):
    df = pd.read_csv(input_path)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{title} Accuracy: {acc:.4f}")
    print()
    print(f"{title} Classification Report:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.show()


def main():
    run_experiment(
        "data/features.csv",
        "results/confusion_matrix.png",
        "Confusion Matrix (Clean Data)"
    )

    print("\n" + "=" * 50 + "\n")

    run_experiment(
        "data/features_noisy.csv",
        "results/confusion_matrix_noisy.png",
        "Confusion Matrix (Noisy Data)"
    )


if __name__ == "__main__":
    main()
