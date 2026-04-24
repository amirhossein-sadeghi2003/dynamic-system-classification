from generate_data import main as generate_data_main
from extract_features import main as extract_features_main
from train_model import main as train_model_main


def main():
    print("Step 1: Generating simulation data...")
    generate_data_main()

    print("\nStep 2: Extracting features...")
    extract_features_main()

    print("\nStep 3: Training classifier...")
    train_model_main()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
