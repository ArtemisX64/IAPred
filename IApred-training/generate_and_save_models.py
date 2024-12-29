import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import joblib
import logging
import argparse
import os
from functions_for_training import (
    read_multiple_fasta,
    sequences_to_vectors,
    remove_constant_features
)

# Set up logging
logging.basicConfig(filename='experiment.log', level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def train_and_save_model(X, y, feature_names, k=119, C=1, gamma=0.01):
    print("\nTraining model...")
    print(f"Parameters: k={k}, C={C}, gamma={gamma}")
    ensure_dir_exists('models')
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    variance_selector = VarianceThreshold(threshold=0)
    X_train_var = variance_selector.fit_transform(X_train)
    X_val_var = variance_selector.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_var)
    X_val_scaled = scaler.transform(X_val_var)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
    selected_feature_mask = selector.get_support()
    selected_feature_names = [name for name, selected in zip(feature_names, selected_feature_mask) if selected]

    model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(X_train_selected, y_train_resampled)

    print("\nSaving model and components...")
    joblib.dump(model, os.path.join('models', 'IApred_SVM.joblib'))
    joblib.dump(scaler, os.path.join('models', 'IApred_scaler.joblib'))
    joblib.dump(variance_selector, os.path.join('models', 'IApred_variance_selector.joblib'))
    joblib.dump(selector, os.path.join('models', 'IApred_feature_selector.joblib'))
    joblib.dump(selected_feature_mask, os.path.join('models', 'IApred_feature_mask.joblib'))
    joblib.dump(feature_names, os.path.join('models', 'IApred_all_feature_names.joblib'))

    print("Files saved in 'models' directory:")
    print("- IApred_SVM.joblib")
    print("- IApred_scaler.joblib")
    print("- IApred_variance_selector.joblib")
    print("- IApred_feature_selector.joblib")
    print("- IApred_feature_mask.joblib")
    print("- IApred_all_feature_names.joblib")

def main():
    parser = argparse.ArgumentParser(description='Train model with 10-fold cross-validation')
    parser.add_argument('--k', type=int, default=529,
                      help='Number of features to select (default: 529)')
    parser.add_argument('--c', type=float, default=1,
                      help='SVM C parameter (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.01,
                      help='SVM gamma parameter (default: 0.01)')
    args = parser.parse_args()


    antigens_dir = "antigens"
    non_antigens_dir = "non-antigens"
    
    antigen_files = [os.path.join(antigens_dir, f) for f in os.listdir(antigens_dir) if f.endswith('.fasta')]
    non_antigen_files = [os.path.join(non_antigens_dir, f) for f in os.listdir(non_antigens_dir) if f.endswith('.fasta')]

    print("Reading sequences...")
    print("Reading antigen files...")
    antigens = read_multiple_fasta(antigen_files)
    print(f"Total sequences in antigens: {len(antigens)}")

    print("\nReading non-antigen files...")
    non_antigens = read_multiple_fasta(non_antigen_files)
    print(f"Total sequences in non-antigens: {len(non_antigens)}")

    all_sequences = antigens + non_antigens
    labels = np.array(['antigen'] * len(antigens) + ['non-antigen'] * len(non_antigens))

    if len(all_sequences) == 0:
        print("Error: No sequences were loaded")
        return

    print("\nExtracting features...")
    X, feature_names, failed_indices = sequences_to_vectors(all_sequences)

    if len(failed_indices) > 0:
        failed_indices = failed_indices.astype(int)
        labels = np.delete(labels, failed_indices)

    print("Filtering constant features...")
    X_filtered, feature_mask, feature_names_filtered = remove_constant_features(X, feature_names)
    print(f"Removed {len(feature_names) - len(feature_names_filtered)} constant features")

    train_and_save_model(
        X_filtered, labels, feature_names_filtered, 
        k=args.k, C=args.c, gamma=args.gamma
    )

    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()
