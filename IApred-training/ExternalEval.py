import numpy as np
import os
from joblib import load
import logging
import pandas as pd
from scipy import stats
from functions_for_training import (
    read_fasta,
    sequences_to_vectors
)


logging.basicConfig(level=logging.INFO)

def calculate_score_stats(scores_array):
    total = len(scores_array)
    pos_count = np.sum(scores_array >= 0)
    neg_count = np.sum(scores_array < 0)
    pos_percentage = (pos_count / total) * 100 if total > 0 else 0

    return total, pos_count, neg_count, pos_percentage

def perform_statistical_tests(data_dict):
    symbols = []
    # Test antigens vs non-antigens
    if stats.mannwhitneyu(data_dict['antigens']['data'],
                         data_dict['non']['data'],
                         alternative='two-sided').pvalue < 0.05:
        symbols.append('@')

    # Test antigens vs proteome
    if stats.mannwhitneyu(data_dict['antigens']['data'],
                         data_dict['proteome']['data'],
                         alternative='two-sided').pvalue < 0.05:
        symbols.append('#')

    # Test non-antigens vs proteome
    if stats.mannwhitneyu(data_dict['non']['data'],
                         data_dict['proteome']['data'],
                         alternative='two-sided').pvalue < 0.05:
        symbols.append('$')

    return ''.join(symbols)

def create_statistics_table(score_data, file_names):
    pathogens = {}
    for scores, filename in zip(score_data, file_names):
        pathogen = '_'.join(filename.split('_')[:2])
        category = filename.split('_')[2].replace('.fasta', '')

        if pathogen not in pathogens:
            pathogens[pathogen] = {}

        scores_array = -np.array(scores)

        # Calculate statistics
        total, pos_count, neg_count, pos_percentage = calculate_score_stats(scores_array)

        pathogens[pathogen][category] = {
            'mean': np.mean(scores_array),
            'std': np.std(scores_array),
            'stats': (total, pos_count, neg_count, pos_percentage),
            'data': scores_array  # Store the actual data for statistical testing
        }


    data = []
    for pathogen in sorted(pathogens.keys()):
        stat_symbols = perform_statistical_tests(pathogens[pathogen])

        row = {
            'Pathogen': pathogen,
            'Antigens': f"{pathogens[pathogen]['antigens']['mean']:.2f} ± {pathogens[pathogen]['antigens']['std']:.2f} ({pathogens[pathogen]['antigens']['stats'][0]}, {pathogens[pathogen]['antigens']['stats'][1]}, {pathogens[pathogen]['antigens']['stats'][2]}, {pathogens[pathogen]['antigens']['stats'][3]:.1f}%)",
            'Non-antigens': f"{pathogens[pathogen]['non']['mean']:.2f} ± {pathogens[pathogen]['non']['std']:.2f} ({pathogens[pathogen]['non']['stats'][0]}, {pathogens[pathogen]['non']['stats'][1]}, {pathogens[pathogen]['non']['stats'][2]}, {pathogens[pathogen]['non']['stats'][3]:.1f}%)",
            'Proteome': f"{pathogens[pathogen]['proteome']['mean']:.2f} ± {pathogens[pathogen]['proteome']['std']:.2f} ({pathogens[pathogen]['proteome']['stats'][0]}, {pathogens[pathogen]['proteome']['stats'][1]}, {pathogens[pathogen]['proteome']['stats'][2]}, {pathogens[pathogen]['proteome']['stats'][3]:.1f}%)",
            'Statistics': stat_symbols
        }
        data.append(row)

    return pd.DataFrame(data)

def process_new_sequences(model, scaler, variance_selector, feature_selector, new_sequence_files, feature_mask, all_feature_names):
    raw_scores = []
    valid_file_names = []

    for filename in new_sequence_files:
        filepath = os.path.join('ExternalEvaluation', filename)
        if os.path.exists(filepath):
            try:
                logging.info(f"Processing file: {filepath}")
                sequences = read_fasta(filepath)
                if not sequences:
                    logging.warning(f"No valid sequences found in {filepath}")
                    continue
                X_new, feature_names, failed_indices = sequences_to_vectors(sequences)

                if X_new is None or len(X_new) == 0:
                    logging.warning(f"No valid feature vectors extracted from {filepath}")
                    continue
                logging.info(f"Extracted features from {len(sequences)} sequences in {filepath}")
                feature_map = {name: i for i, name in enumerate(feature_names)}
                X_new_aligned = np.zeros((X_new.shape[0], len(all_feature_names)))
                for i, feature in enumerate(all_feature_names):
                    if feature in feature_map:
                        X_new_aligned[:, i] = X_new[:, feature_map[feature]]
                X_new_aligned = np.nan_to_num(X_new_aligned, nan=0.0,
                                            posinf=np.finfo(np.float64).max,
                                            neginf=np.finfo(np.float64).min)
                X_new_aligned = np.clip(X_new_aligned, -1e6, 1e6)
                X_new_var_selected = variance_selector.transform(X_new_aligned)
                X_new_scaled = scaler.transform(X_new_var_selected)
                X_new_selected = feature_selector.transform(X_new_scaled)

                scores = model.decision_function(X_new_selected)
                raw_scores.append(scores)
                valid_file_names.append(filename)

                logging.info(f"Successfully processed {filepath}: {len(scores)} predictions generated")

            except Exception as e:
                logging.error(f"Error processing file {filepath}: {str(e)}")
                continue
        else:
            logging.error(f"File not found: {filepath}")

    return raw_scores, valid_file_names

def main():
    try:
        models_dir = 'models'
        svm_model = load(os.path.join(models_dir, 'IApred_SVM.joblib'))
        scaler = load(os.path.join(models_dir, 'IApred_scaler.joblib'))
        variance_selector = load(os.path.join(models_dir, 'IApred_variance_selector.joblib'))
        feature_selector = load(os.path.join(models_dir, 'IApred_feature_selector.joblib'))
        feature_mask = load(os.path.join(models_dir, 'IApred_feature_mask.joblib'))
        all_feature_names = load(os.path.join(models_dir, 'IApred_all_feature_names.joblib'))

        logging.info("Loaded model information")

        if len(feature_mask) != len(all_feature_names):
            logging.warning("Feature mask length mismatch. Adjusting feature mask.")
            feature_mask = np.ones(len(all_feature_names), dtype=bool)

        pathogens = [
            "A_fumigatus", "A_pleuropneumoniae", "A_suum", "B_anthracis", "B_pertussis",
            "C_albicans", "C_gattii", "C_neoformans", "C_pneumoniae", "C_posadasii",
            "C_pseudotuberculosis", "E_granulosus", "F_hepatica", "H_capsulatum", 
            "M_bovis", "P_brasiliensis", "P_vivax", "S_aureus", "S_mansoni",
            "S_pseudintermedius", "T_cruzi", "T_gondii", "T_marneffei", "T_spp",
            "T_whipplei"
        ]

        new_sequence_files = []
        for pathogen in pathogens:
            for category in ['antigens', 'non_antigens', 'proteome']:
                new_sequence_files.append(f"{pathogen}_{category}.fasta")

        # Process sequences
        svm_raw_scores, valid_file_names = process_new_sequences(
            svm_model, scaler, variance_selector, feature_selector,
            new_sequence_files, feature_mask, all_feature_names
        )

        if not svm_raw_scores:
            logging.error("No valid scores were generated.")
            return

        # Generate and display table
        stats_table = create_statistics_table(svm_raw_scores, valid_file_names)

        # Add legends
        stats_legend = """
Legend for Statistics column:
@ - Significant difference between antigens and non-antigens (p < 0.05)
# - Significant difference between antigens and proteome (p < 0.05)
$ - Significant difference between non-antigens and proteome (p < 0.05)
"""

        format_legend = """
Values format: mean ± std (n, x, y, z)
where:
n - total number of analyzed proteins
x - number of proteins with score ≥ 0
y - number of proteins with score < 0
z - percentage of proteins with score ≥ 0
"""

        print("\nIntrinsic Antigenicity Statistics:")
        print(stats_table.to_string(index=False))
        print(stats_legend)
        print(format_legend)

        # Save to CSV with legends
        with open('ExternalEvaluation.csv', 'w') as f:  
            stats_table.to_csv(f, index=False)
            f.write('\n' + stats_legend)
            f.write('\n' + format_legend)

    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}")
        logging.error("Stack trace:", exc_info=True)

if __name__ == "__main__":
    main()
