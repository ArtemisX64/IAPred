import os
import argparse
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from Bio import SeqIO
from functions_for_training import sequences_to_vectors, remove_constant_features

def read_multiple_fasta(file_list):
    sequences = []
    sequence_ids = []

    for file_path in file_list:
        try:
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append(str(record.seq))
                sequence_ids.append(f"{file_path}|{record.id}")
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")

    return sequences, sequence_ids

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_pathogen_splits(category_a_files, category_b_files):
    always_include = [
        'antigensP.fasta', 
        'non-antigensP.fasta',
        'nr_virus_antigens.fasta',
        'nr_virus_non-antigens.fasta'
    ]
    pathogens = []
    for file in category_a_files:
        base_name = os.path.basename(file)  # Extract filename from full path
        if base_name not in always_include:
            pathogen = base_name.replace('_antigens.fasta', '')
            pathogens.append(pathogen)

    # Remove duplicates and sort for consistency
    pathogens = sorted(list(set(pathogens)))


    splits = []
    for test_pathogen in pathogens:
        test_files = [
            os.path.join("antigens", f"{test_pathogen}_antigens.fasta"),
            os.path.join("non-antigens", f"{test_pathogen}_non-antigens.fasta")
        ]

        train_files = {
            'category_a': [f for f in category_a_files if f not in test_files],
            'category_b': [f for f in category_b_files if f not in test_files]
        }

        splits.append((train_files, test_files))

    return splits

def cross_validate_model_lopo(X, y, model, category_a_files, category_b_files, sequence_ids):
    splits = get_pathogen_splits(category_a_files, category_b_files)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)


    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    id_to_idx = {seq_id: idx for idx, seq_id in enumerate(sequence_ids)}

    for train_files, test_files in splits:
        train_indices = []
        test_indices = []

        for idx, seq_id in enumerate(sequence_ids):
            for test_file in test_files:
                pathogen = test_file.replace('_antigens.fasta', '').replace('_non-antigens.fasta', '')
                if pathogen in seq_id:
                    test_indices.append(idx)
                    break
            else:  # If no break occurred (sequence doesn't belong to test pathogen)
                train_indices.append(idx)


        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        model.fit(X[train_indices], y[train_indices])
        y_pred_proba = model.predict_proba(X[test_indices])[:, 1]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y[test_indices], y_pred_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Plot ROC curve for this fold
        pathogen_name = os.path.basename(test_files[0])
        pathogen_name = pathogen_name.replace('_antigens.fasta', '').replace('antigens/', '')
        pathogen_name = pathogen_name.replace('_', '. ')
        ax.plot(fpr, tpr, alpha=0.3, label=f'{pathogen_name} (AUC = {roc_auc:.2f})')

    # Plot mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f ± %0.2f)' % (mean_auc, std_auc),
            lw=2)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)


    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('Leave-One-Pathogen-Out Cross-Validation', fontsize=16, fontweight='bold', pad=20)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  
    ax.legend(bbox_to_anchor=(1.02, 1.0),  
             loc='upper left',
             fontsize=8,
             borderaxespad=0.,
             frameon=True,
             facecolor='white',
             edgecolor='gray',
             framealpha=0.8)

 
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Save the plot 
    ensure_dir_exists('TrainingResults')
    plt.savefig(os.path.join('TrainingResults', 'lopo_cv_roc.png'), bbox_inches='tight', dpi=300)
    print("ROC curve has been saved in TrainingResults/lopo_cv_roc.png")
    plt.close()

    return mean_auc, std_auc, aucs

def main():
    parser = argparse.ArgumentParser(description='Perform LOPO-CV')
    parser.add_argument('--k', type=int, default=119,
                      help='Number of features to select (default: 119)')
    parser.add_argument('--c', type=float, default=1,
                      help='SVM C parameter (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.01,
                      help='SVM gamma parameter (default: 0.01)')
    args = parser.parse_args()

    SVM_PARAMS = {
        'C': args.c,
        'gamma': args.gamma,
        'kernel': 'rbf',
        'probability': True,
        'random_state': 42
    }


    antigens_dir = "antigens"
    non_antigens_dir = "non-antigens"
    
    category_a_files = [f for f in os.listdir(antigens_dir) if f.endswith('.fasta')]
    category_b_files = [f for f in os.listdir(non_antigens_dir) if f.endswith('.fasta')]

    category_a_files = [os.path.join(antigens_dir, f) for f in category_a_files]
    category_b_files = [os.path.join(non_antigens_dir, f) for f in category_b_files]

    feature_file = 'protein_features.pkl'
    sequences_file = 'sequences.pkl'

    try:
        # Try to load pre-computed features
        if os.path.exists(feature_file):
            print(f"Loading pre-computed features from {feature_file}...")
            with open(feature_file, 'rb') as f:
                data = pickle.load(f)
                X = data['X']
                feature_names = data['feature_names']
                labels = data['labels']
                sequence_ids = data['sequence_ids']
                print(f"Successfully loaded features with shape: {X.shape}")
        else:
            X = None

        # If loading failed or file doesn't exist, process sequences
        if X is None:
            print("Reading sequences from FASTA files...")
            category_a_sequences, category_a_ids = read_multiple_fasta(category_a_files)
            category_b_sequences, category_b_ids = read_multiple_fasta(category_b_files)
            all_sequences = category_a_sequences + category_b_sequences
            sequence_ids = category_a_ids + category_b_ids
            labels = np.array(['antigens'] * len(category_a_sequences) + 
                            ['non-antigens'] * len(category_b_sequences))

            print("Extracting sequence features...")
            X, feature_names, failed_indices = sequences_to_vectors(all_sequences)

            if len(failed_indices) > 0:
                print(f"Removing {len(failed_indices)} sequences that failed feature extraction")
                mask = np.ones(len(labels), dtype=bool)
                mask[failed_indices] = False
                labels = labels[mask]
                sequence_ids = [sid for i, sid in enumerate(sequence_ids) if i not in failed_indices]

            # Save features
            with open(feature_file, 'wb') as f:
                pickle.dump({
                    'X': X,
                    'feature_names': feature_names,
                    'labels': labels,
                    'sequence_ids': sequence_ids
                }, f)


        print("Preparing for model training...")
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)


        print("Filtering constant features...")
        X_filtered, feature_mask, feature_names_filtered = remove_constant_features(X, feature_names)


        print("Performing feature selection...")
        selector = SelectKBest(f_classif, k=args.k)
        X_selected = selector.fit_transform(X_filtered, labels_encoded)
        selected_feature_mask = selector.get_support()
        selected_feature_names = [name for name, selected in zip(feature_names_filtered, selected_feature_mask) if selected]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        print("Training and evaluating model...")
        model = SVC(**SVM_PARAMS)
        mean_auc, std_auc, fold_aucs = cross_validate_model_lopo(
            X_scaled, labels_encoded, model,
            category_a_files, category_b_files, sequence_ids
        )

        print(f"LOPO-CV ROC AUC: {mean_auc:.4f} (±{std_auc:.4f})")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
