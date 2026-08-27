import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extract_features import process_file
from generate_data import (
    CLASS_DAMPING_RANGES,
    CRITICAL_DAMPING_N_S_PER_M,
    build_paired_datasets,
    generate_datasets,
)
from train_model import FEATURE_COLUMNS, load_paired_features, paired_split, run_evaluations


class PipelineTests(unittest.TestCase):
    def test_seeded_generation_is_reproducible_and_paired(self):
        clean_a, noisy_a = build_paired_datasets(samples_per_class=3, seed=42)
        clean_b, noisy_b = build_paired_datasets(samples_per_class=3, seed=42)

        pd.testing.assert_frame_equal(clean_a, clean_b)
        pd.testing.assert_frame_equal(noisy_a, noisy_b)
        pd.testing.assert_series_equal(clean_a["sample_id"], noisy_a["sample_id"])
        pd.testing.assert_series_equal(clean_a["label"], noisy_a["label"])
        pd.testing.assert_series_equal(
            clean_a["damping_n_s_per_m"], noisy_a["damping_n_s_per_m"]
        )
        self.assertFalse(
            np.allclose(
                clean_a.filter(regex=r"^x_").to_numpy(),
                noisy_a.filter(regex=r"^x_").to_numpy(),
            )
        )

    def test_declared_class_bands_have_consistent_semantics(self):
        under_low, under_high = CLASS_DAMPING_RANGES["underdamped"]
        near_low, near_high = CLASS_DAMPING_RANGES["near_critical"]
        over_low, over_high = CLASS_DAMPING_RANGES["overdamped"]

        self.assertLess(under_low, under_high)
        self.assertLess(under_high, CRITICAL_DAMPING_N_S_PER_M)
        self.assertLess(near_low, CRITICAL_DAMPING_N_S_PER_M)
        self.assertGreater(near_high, CRITICAL_DAMPING_N_S_PER_M)
        self.assertGreater(over_low, CRITICAL_DAMPING_N_S_PER_M)
        self.assertLess(over_low, over_high)

    def test_committed_datasets_match_seeded_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            clean_path = directory / "simulation_data.csv"
            noisy_path = directory / "simulation_data_noisy.csv"
            clean_features_path = directory / "features.csv"
            noisy_features_path = directory / "features_noisy.csv"

            generate_datasets(clean_path=clean_path, noisy_path=noisy_path)
            process_file(clean_path, clean_features_path)
            process_file(noisy_path, noisy_features_path)

            for generated, committed in [
                (clean_path, ROOT / "data" / "simulation_data.csv"),
                (noisy_path, ROOT / "data" / "simulation_data_noisy.csv"),
                (clean_features_path, ROOT / "data" / "features.csv"),
                (noisy_features_path, ROOT / "data" / "features_noisy.csv"),
            ]:
                pd.testing.assert_frame_equal(
                    pd.read_csv(generated),
                    pd.read_csv(committed),
                    check_exact=False,
                    rtol=1e-9,
                    atol=1e-11,
                )

    def test_seeded_metrics_match_committed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = run_evaluations(results_dir=directory)
        committed = pd.read_csv(ROOT / "results" / "evaluation_metrics.csv")
        pd.testing.assert_frame_equal(generated, committed, check_exact=False, rtol=1e-12)

    def test_split_is_paired_and_metadata_is_not_a_feature(self):
        clean, noisy = load_paired_features(
            ROOT / "data" / "features.csv", ROOT / "data" / "features_noisy.csv"
        )
        train_ids, test_ids = paired_split(clean["label"])

        self.assertEqual(len(train_ids), 240)
        self.assertEqual(len(test_ids), 60)
        self.assertTrue(set(train_ids).isdisjoint(test_ids))
        self.assertTrue(set(test_ids).issubset(noisy.index))
        self.assertNotIn("damping_n_s_per_m", FEATURE_COLUMNS)
        self.assertNotIn("damping_ratio", FEATURE_COLUMNS)
        self.assertNotIn("sample_id", FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
