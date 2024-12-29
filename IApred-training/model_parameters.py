import numpy as np
import re
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_learning_curve(estimator, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring="roc_auc")

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("ROC AUC Score")
    plt.legend(loc="best")
    plt.title("Learning Curve")
    
    ensure_dir_exists('TrainingResults')
    plt.savefig(os.path.join('TrainingResults', 'learning_curve.png'), dpi=300, bbox_inches='tight')
    #plt.show()

def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(10, 7))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={avg_precision:0.2f}')
    
    ensure_dir_exists('TrainingResults')
    plt.savefig(os.path.join('TrainingResults', 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    #plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages for each row
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percentages = (cm / row_sums * 100).round(1)
    
    # Create annotation text with both count and percentage
    annotations = []
    for i in range(cm.shape[0]):
        row_annotations = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percentages[i, j]
            row_annotations.append(f'{count}\n({percentage}%)')
        annotations.append(row_annotations)
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                annot_kws={'size': 12, 'ha': 'center', 'va': 'center'})
    plt.title('Confusion Matrix')
    plt.ylabel('True Antigens')
    plt.xlabel('Predicted Antigens')
    
    ensure_dir_exists('TrainingResults')
    plt.savefig(os.path.join('TrainingResults', 'confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    #plt.show()

def analyze_errors(model, X, y, feature_names, results_file):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    misclassified = X[y != y_pred]
    misclassified_labels = y[y != y_pred]
    misclassified_proba = y_pred_proba[y != y_pred]

    ensure_dir_exists('TrainingResults')
    with open(os.path.join('TrainingResults', results_file), 'w') as f:
        f.write(f"Number of misclassified samples: {len(misclassified)}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y, y_pred))
        f.write("\n")

        false_positives = misclassified[misclassified_labels == 0]
        false_negatives = misclassified[misclassified_labels == 1]

        f.write("\nFalse Positives Analysis:\n")
        f.write(f"Number of false positives: {len(false_positives)}\n")
        f.write("Average feature values:\n")
        for i, feature in enumerate(feature_names):
            avg_value = false_positives[:, i].mean() if len(false_positives) > 0 else 0
            f.write(f"{feature}: {avg_value:.4f}\n")

        f.write("\nFalse Negatives Analysis:\n")
        f.write(f"Number of false negatives: {len(false_negatives)}\n")
        f.write("Average feature values:\n")
        for i, feature in enumerate(feature_names):
            avg_value = false_negatives[:, i].mean() if len(false_negatives) > 0 else 0
            f.write(f"{feature}: {avg_value:.4f}\n")

def train_and_evaluate_model(X, y, feature_names, k=119, C=1, gamma=0.01):
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
    selected_feature_mask = selector.get_support()
    selected_feature_names = [name for name, selected in zip(feature_names, selected_feature_mask) if selected]

    model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(X_train_selected, y_train_resampled)

    y_pred = model.predict(X_val_selected)
    y_pred_proba = model.predict_proba(X_val_selected)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plot_learning_curve(model, X_train_selected, y_train_resampled)
    plot_precision_recall_curve(y_val, y_pred_proba)
    plot_confusion_matrix(y_val, y_pred)

    analyze_errors(model, X_val_selected, y_val, selected_feature_names, 'model_analysis_results.txt')

    return model, scaler, variance_selector, selector, roc_auc

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SVM model')
    parser.add_argument('--k', type=int, default=119,
                      help='Number of features to select (default: 119)')
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

    X_filtered, feature_mask, feature_names_filtered = remove_constant_features(X, feature_names)

    print("\nTraining model...")
    model, scaler, variance_selector, selector, roc_auc = train_and_evaluate_model(
        X_filtered, labels, feature_names_filtered, 
        k=args.k, C=args.c, gamma=args.gamma
    )

    ensure_dir_exists('TrainingResults')
    with open(os.path.join('TrainingResults', 'model_parameters_results.txt'), 'w') as f:
        f.write("Model Parameters and Results\n")
        f.write("==========================\n")
        f.write(f"Number of features (k): {args.k}\n")
        f.write(f"C parameter: {args.c}\n")
        f.write(f"Gamma parameter: {args.gamma}\n")
        f.write(f"ROC AUC Score: {roc_auc:.4f}\n")

    print("\nAll results have been saved in TrainingResults directory!")
    print(f"ROC AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
