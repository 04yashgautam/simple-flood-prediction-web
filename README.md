# Flood Prediction Using Deep Learning

This repository contains a deep‑learning based flood prediction system developed in Python using Keras / TensorFlow. The model forecasts future flood/discharge levels using historical hydrological and meteorological time-series data.

---

## 📋 Table of Contents

1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Features](#features)  
4. [Prerequisites](#prerequisites)  
5. [Installation & Setup](#installation--setup)  
6. [Usage](#usage)  
7. [Datasets & Preprocessing](#datasets--preprocessing)  
8. [Model Architecture](#model-architecture)  
9. [Training & Evaluation](#training--evaluation)  
10. [Results](#results)  
11. [Customization & Extensions](#customization--extensions)  
12. [Troubleshooting](#troubleshooting)  
13. [Contributing](#contributing)  
14. [License](#license)

---

## Overview

A time‑series forecasting tool using LSTM‑based RNN models to predict river discharge or flood indicators – suitable for short‑term forecasting. Models explored include stacked LSTM, Bi‑LSTM, variants with layer normalization and Leaky ReLU, with hyperparameter tuning via Bayesian optimization.

---

## Features

- Multi‑step flood/discharge prediction from meteorological and hydrological time-series data.  
- Explorations of model variants: stacked LSTM, Bi‑LSTM, DRNN with layer normalization & Leaky ReLU activation.  
- Optional Bayesian or grid-based hyperparameter search.  
- Evaluation metrics include RMSE, MAE, Nash–Sutcliffe Efficiency (NSE).

---

## Prerequisites

- Python 3.8 or newer  
- TensorFlow & Keras  
- NumPy, Pandas, scikit-learn, matplotlib  
- Jupyter Notebook / JupyterLab  

Install with:

```bash
pip install -r requirements.txt
```

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/04yashgautam/Flood-Prediction-using-Deep-Learning.git
   cd Flood-Prediction-using-Deep-Learning
   ```
2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Place your dataset files inside `data/`. Supported formats: CSV, Feather, or NumPy arrays.

---

## Usage

### Exploratory Data Analysis
Open `notebooks/eda.ipynb` and inspect trends and correlations in rainfall, discharge, and other variables.

### Training the Model
```bash
python src/train_model.py   --config configs/lstm_config.yaml   --output-dir models/lstm_experiment
```

### Evaluating Performance
```bash
python src/evaluate.py   --model-path models/lstm_experiment/checkpoint.h5   --test-data data/test_dataset.csv
```

---

## Datasets & Preprocessing

- Dataset should include temporal features: rainfall, water level (discharge), flow rate, etc.  
- Missing value handling, normalization (e.g., `StandardScaler`), and train/test splitting scripts are provided under `src/preprocessing.py`.

---

## Model Architecture

- **Stacked LSTM**: multi-layer LSTM network for sequence forecasting.  
- **Bi-LSTM**: Bidirectional LSTM variant for capturing forward/backward temporal dependencies.  
- Advanced variants include **layer normalization** and **Leaky ReLU** activations to stabilize and speed up learning.

---

## Training & Evaluation

- Training logs include loss curves, validation scores, and early stopping options.  
- Evaluation script outputs performance metrics (RMSE, MAE, NSE).  
- Visualizations (actual vs predicted) are generated automatically or via notebooks.

---

## Results

- Best-performing models typically achieve RMSE and high NSE values (> 0.9), indicating strong predictive power.  
- Stacked LSTM or Bi-LSTM variants consistently outperform simpler baselines.

---

## Customization & Extensions

- **Dataset Flexibility**: Adapt scripts to new data sources (rain gauge, satellite, DEM).  
- **Model Extensions**: Try GRU, hybrid CNN‑RNN models, or attention mechanisms.  
- **Hyperparameter Tuning**: Integrate Bayesian optimization or grid search (e.g., Optuna).  
- **Uncertainty Estimation**: Use Bayesian neural networks or dropout for prediction intervals.

---

## Troubleshooting

- **Data Format Issues**: Double-check headers, missing columns, and date parsing.  
- **Model Divergence**: Adjust learning rate, batch size, or try normalization layers.

---
