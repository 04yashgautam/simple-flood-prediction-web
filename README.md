# 🌊 Simple Flood Prediction Web

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A deep learning–based time-series prediction system to forecast flood or discharge levels using historical hydrological and meteorological data. Built with **Keras / TensorFlow**, featuring **SimpleRNN**.

---

## 📋 Table of Contents

1. 📌[Overview](#overview)
2. ✨[Features](#features)
3. 🛠️[Prerequisites](#prerequisites)
4. ⚙️[Installation & Setup](#installation--setup)
5. 🧪[Training & Evaluation](#training--evaluation)
6. 📈[Results](#results)
7. 🔧[Customization & Extensions](#customization--extensions)
8. 🐛[Troubleshooting](#troubleshooting)

---

## 📌 Overview

A time‑series forecasting tool using **RNN model** to predict river discharge or flood indicators for short-term forecasting. 
---

## ✨ Features

✅ **Multi‑Step Prediction**
✅ **Metrics**: RMSE, MAE, NSE (Nash–Sutcliffe Efficiency)
✅ **End-to-End Workflow**: From preprocessing to evaluation and visualization

---

## 🛠️ Prerequisites

* Python 3.8+
* TensorFlow & Keras
* NumPy, Pandas, scikit-learn, Matplotlib
* Jupyter Notebook / JupyterLab

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/04yashgautam/simple-flood-prediction-web.git
cd simple-flood-prediction-web

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Place your dataset inside the 'data/' folder
```

---

## 🧪 Training & Evaluation

* Monitor training via logs and loss/validation plots
* Evaluation metrics include:

  * Root Mean Square Error (RMSE)
  * Mean Absolute Error (MAE)
  * Nash–Sutcliffe Efficiency (NSE)
* Generates side-by-side actual vs. predicted plots

---

## 📈 Results

* Achieves strong predictive accuracy with NSE > 0.9 in well-tuned configurations
* Stacked/Bi-LSTM models outperform traditional single-layer approaches

---

## 🔧 Customization & Extensions

🛠️ Customize and scale your experiments:

* **Dataset**: Integrate rainfall station, satellite, or elevation data
* **Models**: Try GRU, hybrid CNN-RNN, or attention-based models
* **Search**: Leverage `Optuna` or `Ray Tune` for hyperparameter optimization
* **Uncertainty**: Apply dropout or Bayesian RNN for confidence intervals

---

## 🐛 Troubleshooting

* **Incorrect CSV Format** → Ensure date formats, headers, and required columns exist
* **Model Instability** → Tweak batch size, learning rate, and try layer normalization

---

## 📦 requirements.txt

```txt
tensorflow>=2.5.0
keras>=2.4.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
jupyterlab>=3.0.0
yaml>=5.4.0
optuna>=2.10.0  # optional, for Bayesian tuning
seaborn>=0.11.0
```

---

## 📫 Contact

For questions or suggestions, feel free to connect via [GitHub](https://github.com/04yashgautam)!

---
