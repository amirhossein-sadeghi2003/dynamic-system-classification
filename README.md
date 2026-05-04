# Dynamic System Classification

Machine learning project for classifying the damping behavior of a simulated mass-spring-damper system using extracted response features.

This project connects:

- dynamic system simulation
- signal feature extraction
- supervised machine learning
- clean-vs-noisy data evaluation
- intelligent physical systems

The goal is to show how simulated physical system responses can be converted into labeled datasets and used for classification.

---

## Project Overview

This project simulates a mass-spring-damper system under different damping conditions and trains a machine learning model to classify the system behavior.

The three target classes are:

- `underdamped`
- `critical`
- `overdamped`

The workflow is:

`dynamic system simulation → response signal generation → feature extraction → Random Forest classification → clean/noisy evaluation`

The project also studies how added noise affects classification performance.

---

## Why This Project Matters

Many intelligent physical systems rely on measured response signals to identify system behavior or detect operating conditions.

This project is a compact example of that idea:

- simulate a physical dynamic system
- generate labeled response data
- extract meaningful features from signals
- train a classifier
- evaluate robustness under noisy conditions

It is relevant to:

- machine learning for physical systems
- dynamic system analysis
- condition classification
- signal processing
- control-inspired monitoring
- intelligent sensing systems

---

## System Model

The simulated system is a standard mass-spring-damper model:

```text
m x'' + c x' + k x = 0
```

where:

- `m` is the mass
- `c` is the damping coefficient
- `k` is the spring constant
- `x` is the displacement

Different damping values are used to generate three behavior classes:

| Class | Description |
|---|---|
| `underdamped` | Oscillatory response with decaying amplitude |
| `critical` | Fast return to equilibrium without oscillation |
| `overdamped` | Slow non-oscillatory return to equilibrium |

---

## Project Pipeline

### 1. Response Simulation

The project simulates multiple mass-spring-damper responses for the three damping classes.

Generated datasets:

```text
data/simulation_data.csv
data/simulation_data_noisy.csv
```

The noisy dataset is created by adding noise to the simulated response signals.

---

### 2. Feature Extraction

For each response signal, a compact set of features is extracted.

Extracted features:

- `max_abs`
- `final_abs`
- `mean_abs`
- `std_response`
- `zero_crossings`

Generated feature files:

```text
data/features.csv
data/features_noisy.csv
```

These features are simple but interpretable. For example, `zero_crossings` helps distinguish oscillatory underdamped responses from non-oscillatory critical or overdamped responses.

---

### 3. Classification

A `RandomForestClassifier` is trained to classify the damping behavior.

The model is evaluated on:

- clean simulated feature data
- noisy simulated feature data

Generated result files:

```text
results/confusion_matrix.png
results/confusion_matrix_noisy.png
```

---

## Results

Baseline classification results:

| Dataset | Accuracy |
|---|---:|
| Clean data | 1.00 |
| Noisy data | 0.83 |

The results show that:

- the classifier separates ideal simulated damping classes very well
- noisy response signals make classification more difficult
- simple extracted features still provide useful information under noise
- dynamic system behavior can be represented in a machine-learning-friendly format

---

## Output Figures

### Clean Data Confusion Matrix

The classifier achieves perfect classification on the clean simulated feature dataset.

![Clean Data Confusion Matrix](results/confusion_matrix.png)

### Noisy Data Confusion Matrix

The noisy dataset creates a more difficult classification problem and reduces accuracy.

![Noisy Data Confusion Matrix](results/confusion_matrix_noisy.png)

---

## Repository Structure

```text
dynamic-system-classification/
├── data/
│   ├── features.csv
│   ├── features_noisy.csv
│   ├── simulation_data.csv
│   └── simulation_data_noisy.csv
├── results/
│   ├── confusion_matrix.png
│   └── confusion_matrix_noisy.png
├── src/
│   ├── extract_features.py
│   ├── generate_data.py
│   ├── main.py
│   └── train_model.py
├── requirements.txt
└── README.md
```

---

## Main Files

Important files in this project:

- `src/generate_data.py` — simulates mass-spring-damper response data
- `src/extract_features.py` — extracts features from response signals
- `src/train_model.py` — trains and evaluates the classifier
- `src/main.py` — runs the full pipeline

---

## How to Run

Create and activate a virtual environment:

`python3 -m venv venv`

`source venv/bin/activate`

Install dependencies:

`pip install -r requirements.txt`

Run the full pipeline:

`python src/main.py`

This regenerates the simulated datasets, extracted feature files, and confusion matrix results.

---

## Main Libraries

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `scikit-learn`

---

## Project Role in Portfolio

This project is designed as a bridge between machine learning and dynamic physical systems.

It complements other projects such as:

- DC motor simulation
- Kalman filtering for motor state estimation
- embedded TinyML condition monitoring

Together, these projects support a portfolio direction focused on:

```text
AI / ML for intelligent physical systems
```

---

## Limitations

This project has several limitations:

- the data is fully simulated
- the feature set is simple and hand-designed
- the model is evaluated on generated data rather than real sensor data
- the noise model is simplified
- only one classifier is used in the current version

These limitations are acceptable for a compact educational project, but they leave room for future extensions.

---

## Future Work

Possible next steps:

- add more advanced time-domain features
- compare multiple classifiers
- test stronger and more realistic noise models
- use cross-validation
- apply the same idea to DC motor or pendulum systems
- use real sensor data instead of fully simulated responses
- explore time-series models that operate directly on response signals

---

## Summary

This project demonstrates how dynamic system responses can be transformed into a supervised learning problem.

It shows that:

- simulated physical systems can generate useful labeled datasets
- simple signal features can capture important behavior patterns
- machine learning can classify dynamic regimes
- noise can significantly affect classification performance

Overall, this project is a compact portfolio example of machine learning applied to dynamic physical systems.
