# Deep-Learning-Projects
# Bank Customer Churn Prediction — Artificial Neural Network

A binary classification model built with TensorFlow/Keras to predict whether a bank customer will churn (leave the bank), based on demographic and account data.

## Overview

This project implements a feedforward Artificial Neural Network (ANN) that predicts customer churn from tabular banking data. It covers the full pipeline: preprocessing (encoding, scaling), model training, and evaluation.

## Features

- End-to-end preprocessing pipeline (label encoding, one-hot encoding, feature scaling)
- Feedforward ANN with two hidden layers built in Keras
- Stratified train/test split to preserve class balance
- Evaluation via confusion matrix, accuracy, precision, and recall

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- pandas / NumPy

## Dataset

`Churn_Modelling.csv` — 10,000 bank customer records.

**Input features:** Geography, Gender, Credit Score, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary

**Target:** `Exited` (1 = churned, 0 = retained)

> Note: the target class is imbalanced (~80/20 split). Accuracy alone is not a reliable performance measure here — see Results below.

## Model Architecture

```
Input Layer
   → Dense(6, activation='relu')      # Hidden layer 1
   → Dense(6, activation='relu')      # Hidden layer 2
   → Dense(1, activation='sigmoid')   # Output layer (binary)
```

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Batch size:** 32
- **Epochs:** 100

**requirements.txt**
```
pandas
numpy
tensorflow
scikit-learn
```

## Usage

Place `Churn_Modelling.csv` in the `Data/` directory, then run:

```bash
python churn_ann.py
```

The script preprocesses the data, trains the ANN, and prints the confusion matrix and accuracy on the test set.

## Results

| Metric | Value |
|---|---|
| Accuracy | 86.28% |
| Precision (churn class) | ~75.2% |
| Recall (churn class) | ~48.7% |

**Confusion Matrix:**

|  | Predicted: Stay | Predicted: Churn |
|---|---|---|
| **Actual: Stay** | 1909 | 82 |
| **Actual: Churn** | 261 | 248 |

The model correctly identifies about half of actual churners (recall ≈ 49%). Since missing a churner is typically costlier to a business than a false alarm, next steps would focus on improving recall — e.g. class weighting, threshold tuning, or resampling — rather than chasing accuracy.

## Future Improvements

- Address class imbalance (SMOTE or class weights) to improve recall
- Hyperparameter tuning (layer size, learning rate, epochs)
- Add validation set + early stopping to check for overfitting
- Compare against baseline models (logistic regression, random forest)

