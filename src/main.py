from create_response_gallery import main as create_gallery_main
from extract_features import main as extract_features_main
from generate_data import main as generate_data_main
from train_model import main as train_model_main


def main():
    generate_data_main()
    extract_features_main()
    train_model_main()
    create_gallery_main()
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
