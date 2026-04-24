Dynamic System Classification

This project combines dynamic system simulation and machine learning to classify the behavior of a mass-spring-damper system under different damping conditions.

Description

The project uses a simulated mass-spring-damper system to generate response data for three classes:

underdamped
critically damped
overdamped

From each simulated response, a set of simple features is extracted. These features are then used to train a machine learning classifier that predicts the damping category of the system.

The project also evaluates the effect of measurement noise by comparing classification performance on clean and noisy simulated data.

Goals

The goals of this project are:

simulate dynamic system responses
generate labeled datasets from simulation
extract meaningful features from time-series signals
train a classifier for system behavior classification
compare performance on clean and noisy data
connect physical system modeling with applied machine learning
Dynamic System Model

The system is based on the standard mass-spring-damper equation:

m x'' + c x' + k x = 0

where:

m is the mass
c is the damping coefficient
k is the spring constant
x is the displacement

The project uses:

m = 1.0
k = 10.0
x0 = 1.0
v0 = 0.0

The critical damping value is computed from the system parameters and used to define the damping classes.

Pipeline

The project follows this pipeline.

1. Data Generation

The script simulates multiple system responses for:

underdamped cases
critical cases
overdamped cases

Two datasets are generated:

clean simulated data
noisy simulated data

Saved files:

data/simulation_data.csv
data/simulation_data_noisy.csv
2. Feature Extraction

For each response, the following features are extracted:

max_abs
final_abs
mean_abs
std_response
zero_crossings

Saved files:

data/features.csv
data/features_noisy.csv
3. Classification

A RandomForestClassifier is trained to classify the damping behavior of the system.

The classifier is evaluated on:

clean feature data
noisy feature data

Saved files:

results/confusion_matrix.png
results/confusion_matrix_noisy.png
Results

The project produced the following baseline results:

Clean data accuracy: 1.00
Noisy data accuracy: 0.83

These results show that the classifier performs perfectly on ideal simulated data and remains reasonably robust when noise is added to the signals.

This makes the project a simple example of machine learning applied to dynamic physical systems under both ideal and noisy conditions.

Project Structure
dynamic-system-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
├── results/
├── notebooks/
└── src/
    ├── generate_data.py
    ├── extract_features.py
    ├── train_model.py
    └── main.py
Run

From the project root:

python src/main.py

Main Libraries
numpy
scipy
matplotlib
pandas
scikit-learn
Future Work

Possible next steps:

add more advanced features from the response signal
compare multiple classifiers
study the effect of stronger noise levels
apply the same idea to other dynamic systems such as DC motors
use real sensor data instead of fully simulated signals
