import numpy as np
import re
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import logging
import os
from joblib import Parallel, delayed
from functions_for_training import (
    read_multiple_fasta,
    extract_features,
    sequences_to_vectors,
    remove_constant_features,
    calculate_additional_features,
    calculate_edescriptor_features,
    aa_properties
)

# Set up logging
logging.basicConfig(filename='experiment.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def train_and_evaluate_model(X, y, feature_names, k):
    try:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded,
            test_size=0.2,
            stratify=y_encoded,
            random_state=42
        )

        variance_selector = VarianceThreshold(threshold=0)
        X_train_var_selected = variance_selector.fit_transform(X_train)
        X_val_var_selected = variance_selector.transform(X_val)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_var_selected)
        X_val_scaled = scaler.transform(X_val_var_selected)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        k = min(k, X_train_resampled.shape[1])

        selector = SelectKBest(f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
        X_val_selected = selector.transform(X_val_scaled)

        model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
        model.fit(X_train_selected, y_train_resampled)

        y_pred_proba = model.predict_proba(X_val_selected)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        return roc_auc

    except Exception as e:
        print(f"Error in train_and_evaluate_model: {str(e)}")
        raise

def plot_k_roc_auc(k_values, roc_auc_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, roc_auc_scores, '-')
    plt.xlabel('Number of features (k)')
    plt.ylabel('ROC AUC Score')
    plt.title('ROC AUC Score vs Number of Features')
    plt.grid(True)
    ensure_dir_exists('TrainingResults')
    plt.savefig(os.path.join('TrainingResults', 'Best_k_curve.png'), dpi=300, bbox_inches='tight')
    print("Plot has been saved as 'Best_k_curve.png'")
#   plt.show()


def main():
    antigens_dir = "antigens"
    non_antigens_dir = "non-antigens"
    
    antigen_files = [os.path.join(antigens_dir, f) for f in os.listdir(antigens_dir) if f.endswith('.fasta')]
    non_antigen_files = [os.path.join(non_antigens_dir, f) for f in os.listdir(non_antigens_dir) if f.endswith('.fasta')]

    print("Reading antigen files...")
    antigens = read_multiple_fasta(antigen_files)
    print(f"Total sequences in antigens: {len(antigens)}")

    print("\nReading non-antigen files...")
    non_antigens = read_multiple_fasta(non_antigen_files)
    print(f"Total sequences in non-antigens: {len(non_antigens)}")

    all_sequences = antigens + non_antigens
    labels = np.array(['antigen'] * len(antigens) + ['non-antigen'] * len(non_antigens))

    if len(all_sequences) == 0:
        print("Error: No sequences were loaded. Please check your input files.")
        return

    print(f"\nTotal sequences: {len(all_sequences)}")
    print(f"Category distribution: antigens={len(antigens)}, non-antigens={len(non_antigens)}")

    print("\nShowing First Sequence:")
    for i, seq in enumerate(all_sequences[:1]):
        print(f"\nSequence {i+1}:")
        print(f"Length: {len(seq)}")
        print(f"First 50 characters: {seq[:50]}")

        features, feature_names = extract_features(seq)
        if features is None:
            print("Failed to extract features!")
            print("Attempting to debug why:")

            try:
                analysis = ProteinAnalysis(seq)
                print("ProteinAnalysis succeeded")
            except Exception as e:
                print(f"ProteinAnalysis failed: {str(e)}")

            try:
                additional = calculate_additional_features(seq)
                print("Additional features succeeded")
            except Exception as e:
                print(f"Additional features failed: {str(e)}")

            try:
                edesc = calculate_edescriptor_features(seq, aa_properties)
                print("E-descriptor features succeeded")
            except Exception as e:
                print(f"E-descriptor features failed: {str(e)}")
        else:
            print("Feature extraction succeeded")
            print(f"Number of features: {len(features)}")

    try:
        X, feature_names, failed_indices = sequences_to_vectors(all_sequences)

        print(f"\nFeature extraction results:")
        print(f"Total sequences: {len(all_sequences)}")
        print(f"Successfully processed: {X.shape[0]}")
        print(f"Failed sequences: {len(failed_indices)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Success rate: {(len(all_sequences) - len(failed_indices))/len(all_sequences):.2%}")

        if len(failed_indices) > 0:
            failed_indices = failed_indices.astype(int)
            labels = np.delete(labels, failed_indices)

        if X.shape[0] < 2:
            raise ValueError("Not enough valid sequences for analysis (minimum 2 required)")

        print("\nShape information:")
        print(f"X shape: {X.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label distribution: {np.unique(labels, return_counts=True)}")

        X_filtered, feature_mask, feature_names_filtered = remove_constant_features(X, feature_names)

        print(f"\nAfter removing constant features:")
        print(f"X_filtered shape: {X_filtered.shape}")
        print(f"Number of features retained: {len(feature_names_filtered)}")

        total_features = X_filtered.shape[1]
        k_values = list(range(1, min(201, total_features + 1)))
        if total_features > 200:
            k_values.extend(range(225, total_features + 1, 25))

        roc_auc_scores = []

        for k in k_values:
            print(f"\nEvaluating model with k={k}")
            roc_auc = train_and_evaluate_model(X_filtered, labels, feature_names_filtered, k)
            print(f"ROC AUC: {roc_auc:.4f}")
            roc_auc_scores.append(roc_auc)

        best_k = k_values[np.argmax(roc_auc_scores)]
        best_roc_auc = max(roc_auc_scores)

        print(f"\nBest number of features (k): {best_k}")
        print(f"Best ROC AUC: {best_roc_auc:.4f}")

        plot_k_roc_auc(k_values, roc_auc_scores)
        print("Graph of ROC AUC vs k has been displayed.")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
