# Dynamic System Classification

This project combines dynamic system simulation and machine learning to classify the behavior of a mass-spring-damper system under different damping conditions.

## Description

The project uses a simulated mass-spring-damper system to generate response data for three classes:

- underdamped
- critical
- overdamped

For each simulated response, a set of simple features is extracted. These features are then used to train a machine learning model that predicts the damping category of the system.

The project also studies the effect of noise by comparing classification results on clean and noisy simulated data.

## Goals

The goals of this project are:

- simulate dynamic system responses
- generate labeled datasets
- extract useful features from response signals
- train a classifier for damping type recognition
- compare model performance on clean and noisy data
- connect dynamic systems with basic machine learning

## System Model

The system is based on the standard mass-spring-damper equation:

`m x'' + c x' + k x = 0`

where:

- `m` is the mass
- `c` is the damping coefficient
- `k` is the spring constant
- `x` is the displacement

## Pipeline

### 1. Data Generation

The project simulates multiple responses for:

- underdamped systems
- critically damped systems
- overdamped systems

Two datasets are created:

- clean simulated data
- noisy simulated data

Saved files:

- `data/simulation_data.csv`
- `data/simulation_data_noisy.csv`

### 2. Feature Extraction

For each response, the following features are extracted:

- `max_abs`
- `final_abs`
- `mean_abs`
- `std_response`
- `zero_crossings`

Saved files:

- `data/features.csv`
- `data/features_noisy.csv`

### 3. Classification

A `RandomForestClassifier` is trained to classify the damping behavior of the system.

The model is tested on:

- clean feature data
- noisy feature data

Saved files:

- `results/confusion_matrix.png`
- `results/confusion_matrix_noisy.png`

## Results

The project produced the following baseline results:

| Dataset | Accuracy |
|---|---:|
| Clean data | 1.00 |
| Noisy data | 0.83 |

These results show that the classifier performs very well on ideal simulated data and remains reasonably robust when noise is added to the signals.

This project is a simple example of machine learning applied to dynamic physical systems under both ideal and noisy conditions.

## Main Files

Important files in this project:

- `src/generate_data.py`
- `src/extract_features.py`
- `src/train_model.py`
- `src/main.py`

## How to Run

From the project root, run this command:

`python src/main.py`

## Main Libraries

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `scikit-learn`

## Future Work

Possible next steps:

- add more advanced features from the response signal
- compare multiple classifiers
- study the effect of stronger noise levels
- apply the same idea to other dynamic systems such as DC motors
- use real sensor data instead of fully simulated signals
