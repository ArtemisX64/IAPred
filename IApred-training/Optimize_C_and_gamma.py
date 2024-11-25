import numpy as np
import re
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import logging
import os
import pickle
import argparse
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
logging.basicConfig(filename='experiment.log', level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def plot_param_heatmap(grid_scores, C_range, gamma_range):
    scores = np.array([score for _, _, score in grid_scores]).reshape(len(C_range), len(gamma_range))
    plt.figure(figsize=(12, 8))
    sns.heatmap(scores, xticklabels=gamma_range, yticklabels=C_range, 
                annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Grid Search Scores: C vs gamma parameters')
    plt.xlabel('gamma')
    plt.ylabel('C')
    
    ensure_dir_exists('TrainingResults')
    plt.savefig(os.path.join('TrainingResults', 'C_gamma_heatmap.png'), dpi=300, bbox_inches='tight')
    print("Heatmap has been saved in TrainingResults/C_gamma_heatmap.png")
#    plt.show()

def optimize_parameters(X, y, k=119):
    print(f"Starting parameter optimization with k={k}")
    
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

    k = min(k, X_train_resampled.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
    X_val_selected = selector.transform(X_val_scaled)

    C_range = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
    gamma_range = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    grid = StratifiedKFold(n_splits=5)

    print("Performing grid search...")
    grid_scores = []
    total_combinations = len(C_range) * len(gamma_range)
    current = 0

    for C, gamma in product(C_range, gamma_range):
        current += 1
        print(f"\rProgress: {current}/{total_combinations}", end='')
        
        clf = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
        scores = cross_val_score(clf, X_train_selected, y_train_resampled, 
                               cv=grid, scoring='roc_auc')
        grid_scores.append((C, gamma, scores.mean()))

    print("\nGrid search completed")

    best_score = max(grid_scores, key=lambda x: x[2])
    best_C, best_gamma = best_score[0], best_score[1]

    print(f"\nBest parameters found:")
    print(f"C: {best_C}")
    print(f"gamma: {best_gamma}")
    print(f"Score: {best_score[2]:.4f}")

    plot_param_heatmap(grid_scores, C_range, gamma_range)

    return best_C, best_gamma, best_score[2]

def main():
    parser = argparse.ArgumentParser(description='Optimize SVM parameters')
    parser.add_argument('--k', type=int, default=119,
                      help='Number of features to select (default: 119)')
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

    X_filtered, feature_mask, feature_names_filtered = remove_constant_features(X, feature_names)

    print("\nOptimizing parameters...")
    best_C, best_gamma, best_score = optimize_parameters(X_filtered, labels, k=args.k)

    results = {
        'k': args.k,
        'best_C': best_C,
        'best_gamma': best_gamma,
        'best_score': best_score
    }

    with open('c_gamma_optimization_results.txt', 'w') as f:
        f.write("Parameter Optimization Results\n")
        f.write("============================\n")
        f.write(f"Number of features (k): {args.k}\n")
        f.write(f"Best C: {best_C}\n")
        f.write(f"Best gamma: {best_gamma}\n")
        f.write(f"Best score: {best_score:.4f}\n")

    print("\nResults have been saved to 'c_gamma_optimization_results.txt'")

if __name__ == "__main__":
    main()
