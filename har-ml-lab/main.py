import argparse
from pathlib import Path
import sys
import numpy as np
import requests
import zipfile
import io
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import local modules
from data.loader import load_har_signals, load_har_features
from features import extract_features
from models.classic import LogisticRegressionModel, SVMModel, RandomForestModel
from plots import plot_confusion_matrix
# from report import generate_markdown_report

def download_uci_har_dataset(data_root: Path):
    """
    Automatically download and extract the UCI HAR dataset if not found.
    """
    # Target directory: data/raw
    raw_dir = data_root.parent
    if raw_dir.name == "UCI HAR Dataset": # Handle case .../UCI HAR Dataset/UCI HAR Dataset
        raw_dir = raw_dir.parent
    
    # Expected final dataset path
    dataset_dir = raw_dir / "UCI HAR Dataset"
    
    # Check if key file exists
    if (dataset_dir / "train" / "X_train.txt").exists():
        return dataset_dir

    print(f"\nDataset not found. Preparing to download...")
    print(f"Target path: {dataset_dir}")
    
    # Ensure raw directory exists
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    print(f"Downloading from: {url}")
    print("This may take a while (60MB)...")
    
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        
        # Process zip in memory
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        print("Extracting...")
        z.extractall(raw_dir)
        print("Download and extraction complete!\n")
        
        return dataset_dir
        
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        print("Please try downloading manually: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip")
        print(f"And extract to: {raw_dir}")
        sys.exit(1)

def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="HAR Machine Learning Lab - Experiment Runner")
    
    # 1. Data parameters
    parser.add_argument("--data-dir", type=str, default="data/raw/UCI HAR Dataset",
                        help="Path to the dataset root directory")
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="Validation set size ratio (default: 0.2)")
    
    # 2. Feature parameters
    parser.add_argument("--use-custom-features", action="store_true",
                        help="If set, use custom extracted features (63 dim). Otherwise use official features (561 dim).")
    
    # 3. Model selection
    parser.add_argument("--model", type=str, default="lr", choices=["lr", "svm", "rf"],
                        help="Choose model: lr (Logistic Regression), svm (SVM), rf (Random Forest)")
    
    # 4. Hyperparameters
    parser.add_argument("--C", type=float, default=1.0, help="Regularization strength for LR/SVM")
    parser.add_argument("--rf-trees", type=int, default=100, help="Number of trees for Random Forest")
    
    # 5. Report parameters
    parser.add_argument("--save-plots", action="store_true",
                        help="Whether to save confusion matrix plots and reports")
    parser.add_argument("--report-dir", type=str, default="reports",
                        help="Directory to save reports and plots")
    
    args = parser.parse_args()
    
    # Smart path correction
    project_root = Path(__file__).parent.resolve()
    # Ensure data_dir is relative to script or absolute
    if not Path(args.data_dir).is_absolute():
        data_path = project_root / args.data_dir
    else:
        data_path = Path(args.data_dir)
    
    # --- Auto-Download Logic ---
    if not data_path.exists() or not (data_path / "train").exists():
        # Check nested structure
        potential_path = data_path / "UCI HAR Dataset"
        if potential_path.exists():
             data_path = potential_path
        else:
            # Not found, trigger download
            if "UCI HAR Dataset" in args.data_dir:
                 download_target = data_path
            else:
                 download_target = data_path / "UCI HAR Dataset"
                 
            data_path = download_uci_har_dataset(download_target)
            
    args.data_dir = data_path
    
    return args

def load_and_process_custom_features(data_path, val_size=0.2, random_state=42):
    """Pipeline for Stage 3: Raw Signals -> Feature Extraction"""
    print("Reading raw Inertial Signals...")
    signals_train_full, y_train_full = load_har_signals(data_path, split="train")
    signals_test, y_test = load_har_signals(data_path, split="test")
    
    print("Extracting features (63 dim)...")
    X_train_full = extract_features(signals_train_full)
    X_test = extract_features(signals_test)
    
    print(f"Splitting validation set (ratio: {val_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=y_train_full
    )
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_and_process_official_features(data_path, val_size=0.2, random_state=42):
    """Pipeline for Stage 1/2/4: Official Features"""
    print("Reading official 561-dim features...")
    X_train_full, y_train_full = load_har_features(data_path, split="train")
    X_test, y_test = load_har_features(data_path, split="test")
    
    print(f"Splitting validation set (ratio: {val_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=y_train_full
    )
    
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    args = get_args()
    
    print("="*50)
    print(f"Experiment Configuration:")
    print(f"  Data Path: {args.data_dir}")
    print(f"  Features: {'Custom (63 dim)' if args.use_custom_features else 'Official (561 dim)'}")
    print(f"  Model: {args.model.upper()}")
    if args.save_plots:
        print(f"  [√] Results will be saved to: {args.report_dir}")
    print("="*50)

    try:
        # 1. Prepare Data
        print("\nPreparing data...")
        if args.use_custom_features:
            X_train, y_train, X_val, y_val, X_test, y_test = load_and_process_custom_features(
                args.data_dir, val_size=args.val_size
            )
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = load_and_process_official_features(
                args.data_dir, val_size=args.val_size
            )
        
        # 2. Initialize Model
        print(f"\nInitializing Model: {args.model.upper()} ...")
        if args.model == "lr":
            model = LogisticRegressionModel(C=args.C)
        elif args.model == "svm":
            model = SVMModel(C=args.C, kernel="rbf")
        elif args.model == "rf":
            model = RandomForestModel(n_estimators=args.rf_trees)
        else:
            raise ValueError(f"Unknown model type: {args.model}")
            
        # 3. Train
        print(f"Training (samples: {len(X_train)})...")
        model.fit(X_train, y_train)
        print("Training complete.")
        
        # 4. Evaluate
        print("\nEvaluating...")
        y_pred_val = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred_val)
        
        print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print("\nDetailed Classification Report:")
        report_str = classification_report(y_val, y_pred_val)
        print(report_str)
        
        # Save plots and reports
        if args.save_plots:
            # 1. Ensure directory exists
            project_root = Path(__file__).parent.resolve()
            report_path = project_root / args.report_dir
            report_path.mkdir(exist_ok=True)
            
            # 2. Generate filename
            feature_tag = "custom" if args.use_custom_features else "official"
            cm_filename = f"{args.model}_{feature_tag}_cm.png"
            
            # 3. Plot
            plot_confusion_matrix(
                y_val, 
                y_pred_val, 
                save_path=report_path / cm_filename,
                title=f"Confusion Matrix ({args.model.upper()} - {feature_tag})"
            )
            
            # 4. Generate Markdown Report
            # generate_markdown_report(
            #     save_dir=report_path,
            #     model_name=args.model.upper(),
            #     feature_source="Custom (63-dim)" if args.use_custom_features else "Official (561-dim)",
            #     accuracy=val_acc,
            #     classification_report_str=report_str,
            #     confusion_matrix_filename=cm_filename
            # )
        
        print("\n" + "="*50)
        print("Experiment finished.")
        
    except Exception as e:
        print(f"\n[Error] An exception occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
