# **📈 High‑Frequency ETH‑USD Price Prediction**

### **Hybrid XGBoost–LSTM Model • 92.55% Accuracy**

A high‑frequency cryptocurrency forecasting system that combines deep learning and gradient boosting to predict ETH‑USD price movements with **92.55% directional accuracy**. This project includes full data engineering, hybrid modeling, backtesting, and an interactive Streamlit dashboard for real‑time exploration.

## **🚀 Executive Summary**

This project applies a **hybrid **XGBoost–LSTM** architecture** to forecast highly volatile ETH‑USD price movements using high‑frequency market data. The model achieves **92.55% accuracy**, validated through rigorous backtesting on historical price streams.

## **📊 Key Performance Metrics**

| Metric | Value |

| **Directional Accuracy** | **92.55%** |

| **Model Architecture** | Hybrid XGBoost + LSTM |

| **Backtesting** | Included (``backtesting_summary.csv``) |

| **Deployment** | https://bbgmthpqjuashk6xw5mrkq.streamlit.app/ |


## **🧠 Technical Highlights**

### **📡 High‑Frequency Data Engineering**

Ingested minute‑level ETH‑USD price streams from API sources

Applied rolling‑window feature extraction (returns, volatility, momentum)

Normalized and synchronized irregular time intervals

Engineered lag features for sequence learning

### **🔀 Hybrid Model Architecture**

**XGBoost** for short‑term pattern extraction

**LSTM** for temporal sequence learning

Combined predictions using weighted ensemble logic

### **📉 Backtesting & Validation**

Walk‑forward validation

Historical simulation stored in: _results/backtesting_summary.csv_

Metrics computed on unseen test windows

### **🖥️ Deployment (Streamlit GUI)**

Interactive dashboard for:

Real‑time ETH‑USD predictions

Model comparison

Price trend visualization

Run locally: _streamlit run app.py_


## **📁 Project Structure**

├── data/                       # High-frequency ETH-USD data

├── features/                   # Engineered features & preprocessing scripts

├── models/                     # Saved XGBoost & LSTM models

├── results/

│   ├── backtesting_summary.csv # Validation results

│   └── plots/                  # Forecasting visualizations

├── app.py                      # Streamlit dashboard

├── requirements.txt            # Dependencies

└── README.md                   # Project documentation

## **▶️ Installation & Usage**

### **1. Install dependencies**

pip install -r requirements.txt

### **2. Run the Streamlit app**

streamlit run app.py

### **3. Explore**

View real‑time ETH‑USD predictions

Compare model outputs

Inspect backtesting results


## **🧪 Reproducibility**

All experiments use fixed random seeds

Backtesting windows are logged

Data preprocessing steps are fully deterministic


## **👤 Author**

### **Wande Sosina**

MSc Applied Artificial Intelligence & Data Science
