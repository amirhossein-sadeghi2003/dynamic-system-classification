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


def generate_samples(label, c_values, n_samples):
    rows = []

    for _ in range(n_samples):
        c = np.random.uniform(c_values[0], c_values[1])
        response = simulate_response(c)

        row = {f"x_{i}": response[i] for i in range(len(response))}
        row["label"] = label
        row["c_value"] = c

        rows.append(row)

    return rows


def main():
    c_critical = 2 * np.sqrt(m * k)

    underdamped = generate_samples("underdamped", (0.5, c_critical - 0.5), 100)
    critical = generate_samples("critical", (c_critical - 0.1, c_critical + 0.1), 100)
    overdamped = generate_samples("overdamped", (c_critical + 0.5, c_critical + 4.0), 100)

    all_data = underdamped + critical + overdamped
    df = pd.DataFrame(all_data)

    df.to_csv("data/simulation_data.csv", index=False)
    print("Dataset saved to data/simulation_data.csv")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
