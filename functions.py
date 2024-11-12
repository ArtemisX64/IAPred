#@title definitions for k=120

import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
import re

# Define amino acids constant
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            cleaned_sequence = ''.join(aa for aa in str(record.seq).upper() if aa in amino_acids)
            if cleaned_sequence:
                sequences.append(cleaned_sequence)
    return sequences

def read_multiple_fasta(file_list):
    """
    Read sequences from multiple FASTA files
    """
    all_sequences = []
    for file_name in file_list:
        try:
            sequences = read_fasta(file_name)
            all_sequences.extend(sequences)
            logging.info(f"Successfully read {len(sequences)} sequences from {file_name}")
        except Exception as e:
            logging.error(f"Error reading file {file_name}: {str(e)}")
            print(f"Warning: Could not read file {file_name}")
    return all_sequences 

# Only load the motifs that are actually used in the features
required_motifs = {
    'ELME000002', 'ELME000012', 'ELME000045', 'ELME000053', 'ELME000070',
    'ELME000148', 'ELME000220', 'ELME000231', 'ELME000278', 'ELME000304',
    'ELME000321', 'ELME000351', 'ELME000393', 'ELME000394', 'ELME000395',
    'ELME000441', 'ELME000485', 'ELME000528', 'ELME000548'
}

def load_motifs(file_path):
    motifs = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                pattern, name = line.split('#')
                name = name.strip()
                if name in required_motifs:
                    motifs[name] = pattern.strip()
    return motifs

# Only include the amino acid properties that are needed for E-descriptors
aa_properties = {
    'A': [0.008, 0.134, -0.475, -0.039, 0.181],
    'R': [0.171, -0.361, 0.107, -0.258, -0.364],
    'N': [0.255, 0.038, 0.117, 0.118, -0.055],
    'D': [0.303, -0.057, -0.014, 0.225, 0.156],
    'C': [-0.132, 0.174, 0.070, 0.565, -0.374],
    'Q': [0.149, -0.184, -0.030, 0.035, 0.112],
    'E': [0.221, -0.280, -0.315, 0.157, 0.303],
    'G': [0.218, 0.562, -0.024, 0.018, 0.106],
    'H': [0.023, -0.177, 0.041, 0.280, -0.021],
    'I': [-0.353, 0.071, -0.088, -0.195, -0.107],
    'L': [-0.267, 0.018, -0.265, -0.274, 0.206],
    'K': [0.243, -0.339, -0.044, -0.325, -0.027],
    'M': [-0.239, -0.141, -0.155, 0.321, 0.077],
    'F': [-0.329, -0.023, 0.072, -0.002, 0.208],
    'P': [0.173, 0.286, 0.407, -0.215, 0.384],
    'S': [0.199, 0.238, -0.015, -0.068, -0.196],
    'T': [0.068, 0.147, -0.015, -0.132, -0.274],
    'W': [-0.296, -0.186, 0.389, 0.083, 0.297],
    'Y': [-0.141, -0.057, 0.425, -0.096, -0.091],
    'V': [-0.274, 0.136, -0.187, -0.196, -0.299]
}

def calculate_vector_features_optimized(e_vectors, sequence_length):
    """Optimized vector calculations using scipy's pdist"""
    from scipy.spatial.distance import pdist
    import numpy as np
    
    # Use scipy's pdist for faster distance calculations
    distances = pdist(e_vectors, 'euclidean')
    
    # Only calculate angles for a subset of vectors if sequence is long
    if len(e_vectors) > 100:
        # Take every nth vector to reduce computation
        stride = len(e_vectors) // 100
        e_vectors_subset = e_vectors[::stride]
        norms = np.linalg.norm(e_vectors_subset, axis=1)
        dot_products = np.dot(e_vectors_subset, e_vectors_subset.T)
        cos_angles = dot_products / (norms[:, np.newaxis] * norms)
        np.clip(cos_angles, -1.0, 1.0, out=cos_angles)
        angles = np.arccos(cos_angles[np.triu_indices_from(cos_angles, k=1)])
    else:
        norms = np.linalg.norm(e_vectors, axis=1)
        dot_products = np.dot(e_vectors, e_vectors.T)
        cos_angles = dot_products / (norms[:, np.newaxis] * norms)
        np.clip(cos_angles, -1.0, 1.0, out=cos_angles)
        angles = np.arccos(cos_angles[np.triu_indices_from(cos_angles, k=1)])
    
    # Pre-calculate mean vector for both entropy and magnitude
    mean_vector = np.mean(e_vectors, axis=0)
    
    return {
        'e_avg_angle': np.mean(angles),
        'e_avg_distance': np.mean(distances),
        'e_max_distance': np.max(distances),
        'e_vector_entropy': -np.sum(e_vectors * np.log2(np.abs(e_vectors) + 1e-10)) / sequence_length,
        'e_vector_magnitude': np.linalg.norm(mean_vector)
    }

def calculate_required_features(sequence):
    """Calculate only the features that appear in the feature importance CSV"""
    features = {}
    
    # Basic BioPython features that are used
    try:
        analysis = ProteinAnalysis(sequence)
        features.update({
            'aromaticity': analysis.aromaticity(),
            'beta': analysis.secondary_structure_fraction()[1],
            'coil': analysis.secondary_structure_fraction()[2],
            'gravy': analysis.gravy(),
            'instability_index': analysis.instability_index(),
            'pI': analysis.isoelectric_point(),
        })
    except Exception as e:
        return None

    # Calculate used residue features
    seq_len = len(sequence)
    features.update({
        'large_residues': sum(sequence.count(aa) for aa in 'EFHIKLMQRWY') / seq_len,
        'small_residues': sum(sequence.count(aa) for aa in 'ACDGNPSTV') / seq_len,
        'tiny_residues': sum(sequence.count(aa) for aa in 'ACGST') / seq_len
    })

    # Calculate all required transition frequencies
    residue_types = {
        'basic': 'RK',
        'polar': 'QNHSTYC',
        'non_polar': 'AVLIMFPWG'
    }
    
    # Calculate transitions
    for type1, chars1 in residue_types.items():
        for type2, chars2 in residue_types.items():
            key = f'{type1}_to_{type2}'
            if key in ['basic_to_basic', 'polar_to_polar', 'non_polar_to_non_polar']:
                transitions = sum(1 for i in range(len(sequence)-1) 
                               if sequence[i] in chars1 
                               and sequence[i+1] in chars2) / (len(sequence) - 1)
                features[key] = transitions

    # Calculate sequence entropy and repetitiveness
    aa_freq = Counter(sequence)
    features['sequence_entropy'] = -sum((count/seq_len) * np.log2(count/seq_len) 
                                      for count in aa_freq.values() if count > 0)
    features['sequence_repetitiveness'] = sum(sequence.count(sequence[i:i+3]) 
                                            for i in range(len(sequence)-2)) / seq_len

    # Calculate aliphatic index
    features['aliphatic_index'] = (sequence.count('A') + 2.9 * sequence.count('V') + 
                                  3.9 * (sequence.count('I') + sequence.count('L'))) / seq_len * 100

    # Calculate required k-mer frequencies
    required_kmers = {
        'AA', 'AG', 'AK', 'AN', 'AP', 'AT', 'CL', 'CR', 'DG', 'DK',
        'FC', 'FF', 'FI', 'FL', 'GK', 'GN', 'GT', 'HI', 'HL', 'HP',
        'IC', 'IF', 'II', 'IL', 'IR', 'KA', 'KD', 'KE', 'KG', 'KK',
        'KN', 'KP', 'KT', 'KV', 'LC', 'LF', 'LH', 'LI', 'LL', 'LM',
        'LR', 'MI', 'ML', 'MR', 'NA', 'ND', 'NG', 'NK', 'NN', 'NP',
        'NT', 'PA', 'RL', 'TA', 'TD', 'TE', 'TG', 'TK', 'TN', 'TQ',
        'TT', 'VK', 'VN', 'VT', 'WG', 'YN'
    }
    
    for kmer in required_kmers:
        features[f'kmer_{kmer}'] = sequence.count(kmer)

    # Calculate E-descriptor features
    e_vectors = np.array([aa_properties[aa] for aa in sequence if aa in aa_properties])
    if len(e_vectors) > 0:
        # Basic E-descriptor features
        features.update({
            'avg_e1': np.mean(e_vectors[:, 0]),
            'avg_e2': np.mean(e_vectors[:, 1]),
            'avg_e4': np.mean(e_vectors[:, 2]),
            'sum_e1': np.sum(e_vectors[:, 0]),
            'std_e1': np.std(e_vectors[:, 0]),
            'std_e4': np.std(e_vectors[:, 3]),
            'std_e5': np.std(e_vectors[:, 4])
        })

        # Calculate optimized vector relationships
        vector_features = calculate_vector_features_optimized(e_vectors, seq_len)
        features.update(vector_features)

        # Calculate eigenvalues and related features
        covariance_matrix = np.cov(e_vectors.T)
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)

        features.update({
            'e_eigenvalue_5': sorted_eigenvalues[4] if len(sorted_eigenvalues) > 4 else 0,
            'e_explained_variance_2': explained_variance_ratio[1] if len(explained_variance_ratio) > 1 else 0,
            'e_explained_variance_5': explained_variance_ratio[4] if len(explained_variance_ratio) > 4 else 0,
            'e_sphericity': (np.prod(np.abs(eigenvalues))**(1/5)) / (np.mean(np.abs(eigenvalues)))
        })

        # Calculate weighted vector features
        position_weights = np.linspace(0.5, 1.5, len(sequence))
        weighted_vectors = e_vectors * position_weights[:, np.newaxis]
        weighted_avg = np.mean(weighted_vectors, axis=0)
        
        features.update({
            'e_weighted_avg_1': weighted_avg[0],
            'e_weighted_avg_2': weighted_avg[1],
            'e_weighted_avg_4': weighted_avg[3]
        })

    return features

def extract_features(sequence):
    """Extract only the required features from a single sequence"""
    try:
        sequence = ''.join(aa for aa in sequence if aa in amino_acids)
        if not sequence:
            return None, None

        # Get the basic features
        features = calculate_required_features(sequence)
        if features is None:
            return None, None

        # Add motif features
        motifs = load_motifs('protein_motifs4.txt')
        for motif_name, pattern in motifs.items():
            features[f"motif_{motif_name}"] = len(re.findall(pattern, sequence))

        # Convert to list with consistent ordering
        feature_names = sorted(features.keys())
        feature_values = [features[name] for name in feature_names]

        return feature_values, feature_names

    except Exception as e:
        return None, None

def sequences_to_vectors(sequences):
    """Convert sequences to feature vectors with only the required features"""
    try:
        results = []
        for idx, seq in enumerate(sequences):
            features, names = extract_features(seq)
            if features is not None and names is not None:
                results.append((features, names))

        if not results:
            raise ValueError("No valid feature vectors were extracted")

        # Verify all feature vectors have the same features
        reference_names = results[0][1]
        valid_results = []
        failed_indices = []

        for idx, (features, names) in enumerate(results):
            if names == reference_names:
                valid_results.append(features)
            else:
                failed_indices.append(idx)

        if not valid_results:
            raise ValueError("No sequences with consistent features found")

        feature_matrix = np.array(valid_results)
        return feature_matrix, reference_names, np.array(failed_indices)

    except Exception as e:
        raise


def get_feature_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    return result.importances_mean

def remove_constant_features(X, feature_names):
    """
    Remove constant features with added safety checks
    """
    try:
        selector = VarianceThreshold(threshold=0)
        X_without_constant = selector.fit_transform(X)

        # Get the mask of selected features
        constant_feature_mask = selector.get_support()

        # Ensure feature_names is a list for proper indexing
        feature_names = list(feature_names)

        # Select only the feature names that correspond to non-constant features
        selected_feature_names = [name for name, keep in zip(feature_names, constant_feature_mask) if keep]

        print(f"Removed {len(feature_names) - len(selected_feature_names)} constant features")

        return X_without_constant, constant_feature_mask, selected_feature_names

    except Exception as e:
        print(f"Error in remove_constant_features: {str(e)}")
        raise        

def evaluate_sequence(sequence, model, scaler, variance_selector, feature_selector, feature_mask, all_feature_names):
    """
    Evaluate a single sequence using the trained model with clean output.
    """
    import numpy as np
    from functions import extract_features
    
    try:
        features, feature_names = extract_features(sequence)
        if features is None or feature_names is None:
            raise ValueError("Failed to extract features from the sequence")
        
        # Convert to numpy array
        X = np.array(features).reshape(1, -1)
        
        # Create a mapping of feature names to indices
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        # Create a mask that matches the training feature order
        proper_mask = np.zeros(len(feature_names), dtype=bool)
        for idx, name in enumerate(all_feature_names):
            if name in name_to_idx:
                proper_mask[name_to_idx[name]] = feature_mask[idx]
        
        # Apply transformations
        X_filtered = X[:, proper_mask]
        X_var_selected = variance_selector.transform(X_filtered)
        X_scaled = scaler.transform(X_var_selected)
        X_selected = feature_selector.transform(X_scaled)
        
        # Get the prediction score
        score = model.decision_function(X_selected)[0]
        
        return score
        
    except Exception as e:
        raise ValueError(f"Error processing sequence: {str(e)}")
