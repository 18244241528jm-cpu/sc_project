# HAR Machine Learning Lab 🚀

This is a machine learning practical project based on the **UCI HAR (Human Activity Recognition)** dataset. We built a complete machine learning pipeline to recognize 6 different human activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying).

This project not only uses the dataset's native 561-dimensional features but also implements a feature extraction pipeline **starting from raw Inertial Signals**, and compares the performance of Logistic Regression, SVM, and Random Forest models.

---

## 📂 Project Structure

```text
har-ml-lab/
├── data/
│   ├── loader.py       # Data Loader: Loads TXT files from disk
│   ├── preprocess.py   # Preprocessing Pipeline: Train/Validation split, Standardization
│   ├── features.py     # Feature Engineering: Calculates Mean, Std, etc. from raw signals
│   └── __init__.py     # Constants & Configuration
├── models/
│   └── classic.py      # Model Library: Encapsulates LR, SVM, RF
├── reports/            # (Auto-generated) Stores plots
├── tests/              # Unit Tests
├── plots.py            # Plotting Tools: Confusion Matrix, Comparison Plots
├── main.py             # Commander: CLI Entry Point
└── requirements.txt    # Dependency List
```

---

## 🛠️ Installation & Environment

1.  **Create Virtual Environment**:
    ```bash
    cd har-ml-lab
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup**:
    *   **Auto Download (Recommended)**: Run `python main.py` directly, the program will automatically detect and download the UCI HAR Dataset.
    *   **Manual Download**: Download [UCI HAR Dataset.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip) and unzip to `data/raw/UCI HAR Dataset/`.

---

## 🏃‍♂️ Quick Start

### 1. Run Baseline (Using Official 561-dim features)
This is the simplest mode, running Logistic Regression with official features:
```bash
python main.py --model lr
```
*Expected Accuracy: ~96%*

### 2. Run Advanced Models (SVM / Random Forest)
```bash
python main.py --model svm --C 10
python main.py --model rf --rf-trees 200
```
*Expected Accuracy: ~98%*

### 3. Run Custom Features (Stage 3 Challenge) 🔥
Not using official features, but calculating features from raw waveforms (63-dim):
```bash
python main.py --use-custom-features --model rf
```
*Expected Accuracy: ~97.8% (Amazing efficiency!)*

### 4. Generate Plots 📊
Add `--save-plots` argument, the program will save confusion matrix plots to `reports/` directory:
```bash
python main.py --use-custom-features --model rf --save-plots
```

---

## 🔬 Experiment Results Overview

| Model | Features | Dimensions | Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Official | 561 | 96.1% | Baseline |
| **SVM (RBF)** | Official | 561 | 98.2% | Best Performance |
| **Random Forest** | Official | 561 | 97.5% | Robust |
| **Random Forest** | **Custom** | **63** | **97.8%** | **Highlight: Only 1/9 feature dimensions** |

---

## 🧪 Run Tests

This project includes automated tests to ensure data reading and feature calculation logic are correct:
```bash
pytest tests/
```

---

## 📚 References

1. **UCI HAR Dataset**: Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
   [Link to Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

---

*Project by [https://github.com/18244241528jm-cpu](https://github.com/18244241528jm-cpu), 2025*
