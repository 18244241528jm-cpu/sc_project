# HAR Machine Learning Lab

A machine learning project for the **UCI HAR (Human Activity Recognition)** dataset. This system implements a pipeline to recognize 6 human activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) using smartphone sensor data.

The project supports both the dataset's provided 561-dimensional features and a custom feature extraction pipeline from raw inertial signals. It compares Logistic Regression, SVM, and Random Forest models.

---

## Project Structure

```text
har-ml-lab/
├── data/
│   ├── loader.py       # Loads raw TXT files
│   ├── preprocess.py   # Train/Validation split and standardization
│   ├── features.py     # Feature extraction from raw signals
│   └── __init__.py
├── models/
│   └── classic.py      # LR, SVM, and RF model implementations
├── reports/            # Output directory for plots
├── tests/              # Unit tests
├── plots.py            # Confusion matrix plotting
├── main.py             # CLI entry point
└── requirements.txt    # Python dependencies
```

---

## Installation

1.  **Create Virtual Environment**:
    ```bash
    cd har-ml-lab
    python3 -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup**:
    *   **Auto Download**: Run `python main.py`. The script automatically downloads the UCI HAR Dataset if missing.
    *   **Manual Download**: Place the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip) in `data/raw/UCI HAR Dataset/`.

---

## Usage

### Baseline (Official 561-dim features)
Run Logistic Regression using the pre-computed features:
```bash
python main.py --model lr
```

### Advanced Models
Support for SVM and Random Forest:
```bash
python main.py --model svm --C 10
python main.py --model rf --rf-trees 200
```

### Custom Features (Raw Signals)
Extract 63-dimensional features from raw inertial signals instead of using the provided features:
```bash
python main.py --use-custom-features --model rf
```

### Visualization
Generate confusion matrix plots in the `reports/` directory:
```bash
python main.py --use-custom-features --model rf --save-plots
```

---

## Results Summary

| Model | Features | Dimensions | Accuracy |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Official | 561 | 96.1% |
| **SVM (RBF)** | Official | 561 | 98.2% |
| **Random Forest** | Official | 561 | 97.5% |
| **Random Forest** | Custom | 63 | 97.8% |

---

## Testing

Run unit tests to verify data loading and feature extraction:
```bash
pytest tests/
```

---

## References

1. **UCI HAR Dataset**: Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN 2013.
   [Link to Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
