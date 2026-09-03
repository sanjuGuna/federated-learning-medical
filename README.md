# Federated GAT+RDBN Medical Diagnosis Framework

A Graph Attention Network (GAT) combined with Residual Deep Belief Networks (RDBN) and Federated Learning (FL) for multi-dataset clinical prediction and explainable diagnosis.

## 📌 Features

- **Multi-Dataset Clinical Support**: Supports prediction across three medical datasets:
  - **Diabetes Risk Prediction**: Early-stage diabetes symptom classification.
  - **HCV (Hepatitis C Virus) Prediction**: Stage detection for hepatitis, fibrosis, and cirrhosis.
  - **Dermatology Prediction**: Differential diagnosis of erythematous-squamous diseases.
- **Graph Attention Networks (GAT)**: Models complex feature interactions by constructing k-NN patient similarity graphs.
- **Residual Deep Belief Networks (RDBN)**: Captures hierarchical non-linear feature representations.
- **Federated Learning (FedAvg)**: Enables multi-client collaborative model training without sharing sensitive patient data.
- **Interactive Web Interface & API**: Fast, responsive web UI with real-time explainability (attention weights & feature importances).

---

## 📁 Repository Structure

```
├── archive/                  # Preprocessed raw medical datasets
│   ├── diabetes_data_upload.csv
│   ├── hcvdat0.csv
│   ├── dermatology.data
│   └── dermatology.names
├── frontend/                 # Web UI assets
│   ├── index.html
│   ├── style.css
│   └── script.js
├── saved_models/             # Centralized trained model checkpoints
├── federated_models/         # Federated learning trained model checkpoints
├── results/                  # Training and evaluation metrics (CSVs/CMs)
├── model.py                  # GAT, RDBN, and Hybrid GAT+RDBN architectures
├── graph_builder.py          # k-NN Graph construction module
├── preprocess.py             # Preprocessing pipeline for Diabetes dataset
├── hcv_preprocess.py         # Preprocessing pipeline for HCV dataset
├── dermatology_preprocess.py # Preprocessing pipeline for Dermatology dataset
├── train.py                  # Centralized single-dataset training script
├── federated_train.py        # Federated Learning simulation training script
├── evaluate.py               # Evaluation metric script
├── predict.py                # Single-patient and multi-dataset inference script
├── generate_samples.py       # Diagnostic sample generation utility
├── test_fl_model.py          # Testing script for federated learning checkpoints
├── serve.py                  # FastAPI web server and API endpoints
├── run.sh                    # Automated setup and baseline execution script
├── start_ui.sh               # Quick-start script for launching the Web UI
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/sanjuGuna/federated-learning-medical.git
cd federated-learning-medical
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (Command Prompt):
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Launching the Web UI & API Dashboard
To run the interactive web interface and prediction backend API:
```bash
python serve.py
```
or run the quick startup script:
```bash
chmod +x start_ui.sh
./start_ui.sh
```
Open your browser and navigate to: **`http://localhost:8000`**

### 2. Training Models (Centralized Baseline)
To train centralized GAT, RDBN, and Hybrid models on the datasets:
```bash
python train.py --dataset diabetes
python train.py --dataset hcv
python train.py --dataset dermatology
```

### 3. Training Models (Federated Learning Simulation)
To run multi-institution Federated Averaging (FedAvg) training across clients:
```bash
python federated_train.py --dataset diabetes --clients 3 --rounds 50
python federated_train.py --dataset hcv --clients 3 --rounds 50
python federated_train.py --dataset dermatology --clients 3 --rounds 50
```

### 4. Running Model Evaluation
To evaluate trained models and update metrics CSVs:
```bash
python evaluate.py
```

### 5. Running Single Inference / Sample Predictions
```bash
python predict.py
```

---

## 📜 License
This project is open-source under the MIT License.
