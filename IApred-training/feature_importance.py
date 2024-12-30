import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import os
import argparse
from functions_for_training import (
    read_multiple_fasta,
    sequences_to_vectors,
    remove_constant_features
)

def get_feature_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    raw_importance = result.importances_mean
    
    # Normalize by sum of absolute values to get relative contributions
    total_absolute_importance = np.sum(np.abs(raw_importance))
    normalized_importance = raw_importance / total_absolute_importance
    
    return normalized_importance

def save_importance_scores(feature_names, importance_scores, output_dir):
    inverted_importance_scores = -1 * importance_scores
    results = pd.DataFrame({
        'Number': range(1, len(feature_names) + 1),
        'Feature': feature_names,
        'Importance': inverted_importance_scores
    })
    
    # Verify normalization before saving
    abs_sum = np.sum(np.abs(results['Importance'].values))
    print(f"\nVerification before saving:")
    print(f"Sum of absolute contributions: {abs_sum:.6f}")
    print(f"Min contribution: {results['Importance'].min():.1%}")
    print(f"Max contribution: {results['Importance'].max():.1%}")
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    results.to_csv(csv_path, index=False)
    print(f"\nFeature contributions saved to {csv_path}")
    
    return csv_path

def create_visualization(X_selected, csv_path, output_dir):
    importance_df = pd.read_csv(csv_path)
    
    # Create figure with appropriate dimensions
    fig = plt.figure(figsize=(20, 23))
    gs = fig.add_gridspec(23, 20)
    
    # Plot correlation heatmap
    ax_heatmap = fig.add_subplot(gs[:20, :18])
    corr = np.corrcoef(X_selected.T)
    im = ax_heatmap.imshow(corr, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    
    # Handle feature names
    def shorten_name(name, max_len=30):
        return name if len(name) <= max_len else name[:max_len-3] + '...'
    
    feature_names = importance_df['Feature'].apply(shorten_name).values
    
    # Set heatmap labels only if the number of features is less than 150
    if len(feature_names) < 150:
        ax_heatmap.set_xticks(range(len(feature_names)))
        ax_heatmap.set_yticks(range(len(feature_names)))
        ax_heatmap.set_xticklabels(feature_names, rotation=90, ha='left', fontsize=7)
        ax_heatmap.set_yticklabels(feature_names, fontsize=8)
        ax_heatmap.xaxis.set_ticks_position('top')
    else:
        ax_heatmap.set_xticks([])
        ax_heatmap.set_yticks([])
    
    # Add colorbar for correlation
    cbar_ax = fig.add_subplot(gs[21:22, :18])
    plt.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Correlation')
    
    # Plot feature contributions histogram with inverted values
    ax_hist = fig.add_subplot(gs[:20, 18:])
    contributions = -1 * importance_df['Importance'].values  # Invert the values only for visualization
    
    # Create bars with diverging colors
    colors = ['#4B4BFF' if c > 0 else '#FF4B4B' for c in contributions]
    bars = ax_hist.barh(range(len(contributions)-1, -1, -1), contributions,
                       align='center', height=0.8, color=colors)
    
    # Add light gridlines
    ax_hist.grid(True, axis='x', linestyle='--', alpha=0.2, color='gray')
    
    # Set histogram limits and ticks
    ax_hist.set_ylim(-0.5, len(contributions) - 0.5)
    
    # Set symmetric limits based on max absolute value
    max_abs_value = max(abs(contributions.min()), abs(contributions.max()))
    margin = max_abs_value * 0.05
    ax_hist.set_xlim(-max_abs_value - margin, max_abs_value + margin)
    
    # Format x-axis ticks as percentages
    major_ticks = [-max_abs_value, 0, max_abs_value]
    ax_hist.set_xticks(major_ticks)
    ax_hist.set_xticklabels([f'{x:.1%}' for x in major_ticks])
    
    # Add minor ticks
    minor_ticks = np.linspace(-max_abs_value, max_abs_value, 11)
    ax_hist.set_xticks(minor_ticks, minor=True)
    
    # Remove y-axis ticks
    ax_hist.set_yticks([])
    
    # Add zero line
    ax_hist.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
    
    # Customize spines
    for spine in ax_hist.spines.values():
        spine.set_linewidth(0.5)
    
    # Titles and labels
    plt.suptitle('Feature Correlation', y=1, fontsize=17)
    ax_hist.set_title('Relative\nFeature\nImportance', pad=45, fontsize=12)
    ax_hist.set_xlabel('Contribution to Model', fontsize=10, labelpad=10)
    
    # Print verification info
    abs_sum = np.sum(np.abs(contributions))
    print(f"\nVerification:")
    print(f"Sum of absolute contributions: {abs_sum:.6f}")
    print(f"Range of contributions: [{contributions.min():.1%}, {contributions.max():.1%}]")
    
    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'feature_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {plot_path}")

def analyze_and_select_features(X, y, feature_names, k=119):
    print(f"\nStarting feature analysis with {len(feature_names)} features...")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Balance classes using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    
    # Select top k features
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y_resampled)
    feature_mask = selector.get_support()
    
    # Get selected feature names
    selected_features = [name for name, selected in zip(feature_names, feature_mask) if selected]
    
    print(f"Selected {len(selected_features)} features")
    
    # Train SVM model
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_selected, y_resampled)
    
    # Calculate normalized feature contributions
    importance_scores = get_feature_importance(model, X_selected, y_resampled)
    
    # Verify normalization
    abs_sum = np.sum(np.abs(importance_scores))
    print(f"Sum of absolute contributions: {abs_sum:.6f}")
    
    return X_selected, selected_features, importance_scores


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze feature importance')
    parser.add_argument('--k', type=int, default=529,
                      help='Number of features to select (default: 529)')
    parser.add_argument('--c', type=float, default=1,
                      help='SVM C parameter (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.01,
                      help='SVM gamma parameter (default: 0.01)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = 'TrainingResults'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load sequences
    print("Loading sequences...")
    antigens_dir = "antigens"
    non_antigens_dir = "non-antigens"
    
    antigen_files = [os.path.join(antigens_dir, f) for f in os.listdir(antigens_dir) if f.endswith('.fasta')]
    non_antigen_files = [os.path.join(non_antigens_dir, f) for f in os.listdir(non_antigens_dir) if f.endswith('.fasta')]
    
    antigens = read_multiple_fasta(antigen_files)
    non_antigens = read_multiple_fasta(non_antigen_files)
    
    # Create labels
    all_sequences = antigens + non_antigens
    labels = np.array(['antigen'] * len(antigens) + ['non-antigen'] * len(non_antigens))
    
    if not all_sequences:
        print("Error: No sequences loaded")
        return
    
    # Extract features
    print("Extracting features...")
    X, feature_names, failed_indices = sequences_to_vectors(all_sequences)
    
    if len(failed_indices) > 0:
        labels = np.delete(labels, failed_indices)
    
    # Remove constant features
    X_filtered, _, feature_names_filtered = remove_constant_features(X, feature_names)
    
    # Analyze and select features
    X_selected, selected_features, importance_scores = analyze_and_select_features(
        X_filtered, labels, feature_names_filtered, k=args.k
    )
    
    # Save importance scores
    csv_path = save_importance_scores(selected_features, importance_scores, output_dir)
    
    # Create visualization
    create_visualization(X_selected, csv_path, output_dir)

if __name__ == "__main__":
    main()
