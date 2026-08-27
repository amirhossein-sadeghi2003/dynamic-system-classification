from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MASS_KG = 1.0
SPRING_CONSTANT_N_PER_M = 10.0
INITIAL_DISPLACEMENT_M = 1.0
INITIAL_VELOCITY_M_PER_S = 0.0
TIME_S = np.linspace(0.0, 10.0, 100)
CRITICAL_DAMPING_N_S_PER_M = 2.0 * np.sqrt(MASS_KG * SPRING_CONSTANT_N_PER_M)

CLASS_DAMPING_RANGES = {
    "underdamped": (0.5, CRITICAL_DAMPING_N_S_PER_M - 0.5),
    "near_critical": (
        CRITICAL_DAMPING_N_S_PER_M - 0.1,
        CRITICAL_DAMPING_N_S_PER_M + 0.1,
    ),
    "overdamped": (
        CRITICAL_DAMPING_N_S_PER_M + 0.5,
        CRITICAL_DAMPING_N_S_PER_M + 4.0,
    ),
}


def system(_time, state, damping):
    displacement, velocity = state
    acceleration = (
        -damping * velocity - SPRING_CONSTANT_N_PER_M * displacement
    ) / MASS_KG
    return [velocity, acceleration]


def simulate_response(damping):
    solution = solve_ivp(
        system,
        (TIME_S[0], TIME_S[-1]),
        [INITIAL_DISPLACEMENT_M, INITIAL_VELOCITY_M_PER_S],
        t_eval=TIME_S,
        args=(damping,),
        rtol=1e-9,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0]


def response_row(sample_id, label, damping, response):
    row = {
        "sample_id": sample_id,
        "label": label,
        "damping_n_s_per_m": damping,
        "damping_ratio": damping / CRITICAL_DAMPING_N_S_PER_M,
    }
    row.update({f"x_{index}": value for index, value in enumerate(response)})
    return row


def build_paired_datasets(samples_per_class=100, seed=42, noise_std_m=0.03):
    parameter_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1)
    clean_rows = []
    noisy_rows = []

    for label, damping_range in CLASS_DAMPING_RANGES.items():
        damping_values = parameter_rng.uniform(
            damping_range[0], damping_range[1], size=samples_per_class
        )
        for index, damping in enumerate(damping_values):
            sample_id = f"{label}-{index:03d}"
            clean_response = simulate_response(damping)
            noisy_response = clean_response + noise_rng.normal(
                0.0, noise_std_m, size=clean_response.size
            )
            clean_rows.append(
                response_row(sample_id, label, damping, clean_response)
            )
            noisy_rows.append(
                response_row(sample_id, label, damping, noisy_response)
            )

    return pd.DataFrame(clean_rows), pd.DataFrame(noisy_rows)


def generate_datasets(
    clean_path=DATA_DIR / "simulation_data.csv",
    noisy_path=DATA_DIR / "simulation_data_noisy.csv",
    samples_per_class=100,
    seed=42,
    noise_std_m=0.03,
):
    clean_df, noisy_df = build_paired_datasets(
        samples_per_class=samples_per_class,
        seed=seed,
        noise_std_m=noise_std_m,
    )
    clean_path = Path(clean_path)
    noisy_path = Path(noisy_path)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    noisy_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_path, index=False)
    noisy_df.to_csv(noisy_path, index=False)
    return clean_df, noisy_df


def main():
    clean_df, noisy_df = generate_datasets()
    print(f"Saved {len(clean_df)} paired clean and noisy responses.")


if __name__ == "__main__":
    main()
