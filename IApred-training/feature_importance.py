import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import csv
import argparse
from functions_for_training import (
    read_multiple_fasta,
    sequences_to_vectors,
    remove_constant_features,
    get_feature_importance
)

# Set up logging
logging.basicConfig(filename='experiment.log', level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def plot_feature_correlation_with_importance(X, feature_names, importances, sort=False):
    if sort:
        sorted_indices = np.argsort(importances)[::-1]
        feature_names = [feature_names[i] for i in sorted_indices]
        importances = importances[sorted_indices]
        X = X[:, sorted_indices]

    corr = np.corrcoef(X.T)

    fig = plt.figure(figsize=(20, 23))
    gs = fig.add_gridspec(23, 20)

    ax_heatmap = fig.add_subplot(gs[:20, :18])

    im = ax_heatmap.imshow(corr, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)

    def shorten_feature_name(name, max_length=30):
        return name[:max_length-3] + '...' if len(name) > max_length else name

    shortened_labels = [shorten_feature_name(name) for name in feature_names]

    ax_heatmap.set_yticks(range(len(feature_names)))
    ax_heatmap.set_yticklabels(shortened_labels, fontsize=8)

    ax_heatmap.set_xticks(range(len(feature_names)))
    ax_heatmap.set_xticklabels(shortened_labels, rotation=90, ha='left', fontsize=7)

    ax_heatmap.xaxis.set_ticks_position('top')
    ax_heatmap.xaxis.set_label_position('top')

    ax_hist = fig.add_subplot(gs[:20, 18:])
    ax_hist.barh(range(len(importances)), importances, align='edge', height=1)
    ax_hist.set_ylim(0, len(importances))

    max_importance = max(abs(min(importances)), abs(max(importances)))
    ax_hist.set_xlim(-max_importance, max_importance)
    ax_hist.set_xticks([-max_importance, 0, max_importance])
    ax_hist.set_xticklabels([f'{-max_importance:.3f}', '0', f'{max_importance:.3f}'])
    ax_hist.set_yticks([])

    cbar_ax = fig.add_subplot(gs[21:22, :18])
    plt.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Correlation')

    plt.suptitle('Feature Correlation and Importance Analysis', y=0.95, fontsize=17)
    ax_hist.set_title('Feature \nImportance', pad=45, fontsize=12)

    plt.subplots_adjust(hspace=0.3)

    ensure_dir_exists('TrainingResults')
    plt.savefig(os.path.join('TrainingResults', 'feature_correlation_importance.png'), dpi=300, bbox_inches='tight')
    print("Plot has been saved in TrainingResults/feature_correlation_importance.png")
#    plt.show()

def save_feature_importances_to_csv(feature_names, importances, filename, sort=False):
    if sort:
        sorted_indices = np.argsort(importances)[::-1]
        feature_names = [feature_names[i] for i in sorted_indices]
        importances = importances[sorted_indices]

    ensure_dir_exists('TrainingResults')
    filepath = os.path.join('TrainingResults', filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Number', 'Feature', 'Importance'])
        for i, (feature, importance) in enumerate(zip(feature_names, importances), 1):
            writer.writerow([i, feature, importance])
    print(f"Feature importances have been saved to '{filepath}'")

def analyze_features(X, y, feature_names, k=119, C=1, gamma=0.01):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

    variance_selector = VarianceThreshold(threshold=0)
    X_var_selected = variance_selector.fit_transform(X_resampled)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_var_selected)

    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y_resampled)
    selected_feature_mask = selector.get_support()
    selected_feature_names = [name for name, selected in zip(feature_names, selected_feature_mask) if selected]

    model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(X_selected, y_resampled)

    importance = get_feature_importance(model, X_selected, y_resampled)

    return X_selected, selected_feature_names, importance

def main():
    parser = argparse.ArgumentParser(description='Analyze feature importance')
    parser.add_argument('--k', type=int, default=119,
                      help='Number of features to select (default: 119)')
    parser.add_argument('--c', type=float, default=1,
                      help='SVM C parameter (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.01,
                      help='SVM gamma parameter (default: 0.01)')
    parser.add_argument('--sort', action='store_true',
                      help='Sort features by importance (default: False)')
    args = parser.parse_args()


    antigens_dir = "antigens"
    non_antigens_dir = "non-antigens"
    
    antigen_files = [os.path.join(antigens_dir, f) for f in os.listdir(antigens_dir) if f.endswith('.fasta')]
    non_antigen_files = [os.path.join(non_antigens_dir, f) for f in os.listdir(non_antigens_dir) if f.endswith('.fasta')]

    print("Reading sequences...")
    antigens = read_multiple_fasta(antigen_files)
    non_antigens = read_multiple_fasta(non_antigen_files)
    
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
    print(f"Removed {len(feature_names) - len(feature_names_filtered)} constant features")

    print("\nAnalyzing features...")
    X_selected, feature_names, importance = analyze_features(
        X_filtered, labels, feature_names_filtered, 
        k=args.k, C=args.c, gamma=args.gamma
    )

    plot_feature_correlation_with_importance(X_selected, feature_names, importance, sort=args.sort)
    save_feature_importances_to_csv(feature_names, importance, 'feature_importances.csv', sort=args.sort)

if __name__ == "__main__":
    main()
