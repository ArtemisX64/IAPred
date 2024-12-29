import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
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

def perform_cross_validation(X, y, k=119, C=1, gamma=0.01, cv=10):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))

    for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y_encoded), 1):
        print(f"\nProcessing fold {fold}/{cv}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        variance_selector = VarianceThreshold(threshold=0)
        X_train_var = variance_selector.fit_transform(X_train)
        X_test_var = variance_selector.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_var)
        X_test_scaled = scaler.transform(X_test_var)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        selector = SelectKBest(f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
        X_test_selected = selector.transform(X_test_scaled)

        model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
        model.fit(X_train_selected, y_train_resampled)
        
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold} ROC (AUC = {roc_auc:.2f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f ± %0.2f)' % (mean_auc, std_auc),
             lw=2)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'±1 std. dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('10-Fold Cross-Validation ROC Curves\n'
             f'k={k}, C={C}, gamma={gamma}',
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right')
    
    ensure_dir_exists('TrainingResults')
    plt.savefig(os.path.join('TrainingResults', '10fold_cv_roc.png'), dpi=300, bbox_inches='tight')
    print("\nROC curve has been saved in TrainingResults/10fold_cv_roc.png")
#    plt.show()

    return mean_auc, std_auc

def main():
    parser = argparse.ArgumentParser(description='Perform 10-fold cross-validation')
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

    print("\nPerforming 10-fold cross-validation...")
    mean_auc, std_auc = perform_cross_validation(
        X_filtered, labels, 
        k=args.k, C=args.c, gamma=args.gamma
    )

    ensure_dir_exists('TrainingResults')
    with open(os.path.join('TrainingResults', 'cross_validation_results.txt'), 'w') as f:
        f.write("10-Fold Cross-Validation Results\n")
        f.write("===============================\n")
        f.write(f"Parameters:\n")
        f.write(f"k: {args.k}\n")
        f.write(f"C: {args.c}\n")
        f.write(f"gamma: {args.gamma}\n\n")
        f.write(f"Results:\n")
        f.write(f"Mean ROC AUC: {mean_auc:.4f}\n")
        f.write(f"Standard Deviation: {std_auc:.4f}\n")

    print("\nProcess completed successfully!")
    print(f"Mean ROC AUC: {mean_auc:.4f} (±{std_auc:.4f})")
    print("Results have been saved in TrainingResults/cross_validation_results.txt")

if __name__ == "__main__":
    main()
