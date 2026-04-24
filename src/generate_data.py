import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


m = 1.0
k = 10.0
x0 = 1.0
v0 = 0.0

t = np.linspace(0, 10, 100)


def system(time, y, c):
    x = y[0]
    v = y[1]

    dxdt = v
    dvdt = -(c / m) * v - (k / m) * x

    return [dxdt, dvdt]


def simulate_response(c):
    sol = solve_ivp(system, [0, 10], [x0, v0], t_eval=t, args=(c,))
    return sol.y[0]


def add_noise(signal, noise_std=0.03):
    noise = np.random.normal(0, noise_std, size=len(signal))
    return signal + noise


def generate_samples(label, c_values, n_samples, noisy=False):
    rows = []

    for _ in range(n_samples):
        c = np.random.uniform(c_values[0], c_values[1])
        response = simulate_response(c)

        if noisy:
            response = add_noise(response)

        row = {f"x_{i}": response[i] for i in range(len(response))}
        row["label"] = label
        row["c_value"] = c

        rows.append(row)

    return rows


def build_dataset(noisy=False):
    c_critical = 2 * np.sqrt(m * k)

    underdamped = generate_samples("underdamped", (0.5, c_critical - 0.5), 100, noisy=noisy)
    critical = generate_samples("critical", (c_critical - 0.1, c_critical + 0.1), 100, noisy=noisy)
    overdamped = generate_samples("overdamped", (c_critical + 0.5, c_critical + 4.0), 100, noisy=noisy)

    all_data = underdamped + critical + overdamped
    return pd.DataFrame(all_data)


def main():
    clean_df = build_dataset(noisy=False)
    noisy_df = build_dataset(noisy=True)

    clean_df.to_csv("data/simulation_data.csv", index=False)
    noisy_df.to_csv("data/simulation_data_noisy.csv", index=False)

    print("Clean dataset saved to data/simulation_data.csv")
    print("Noisy dataset saved to data/simulation_data_noisy.csv")
    print()
    print("Clean label counts:")
    print(clean_df["label"].value_counts())
    print()
    print("Noisy label counts:")
    print(noisy_df["label"].value_counts())


if __name__ == "__main__":
    main()
