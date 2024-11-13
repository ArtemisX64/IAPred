import sys
import os
import argparse
from joblib import load
import numpy as np
from Bio import SeqIO
from functions import evaluate_sequence
import csv

def check_and_install_dependencies():
    required_modules = ['numpy', 'Bio', 'sklearn', 'joblib']
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            print(f"{module} is not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", module])
            print(f"{module} has been installed.")

    print("All required modules are now installed.")

def create_ascii_visualization(score):
    """
    Create an ASCII visualization with indicator position based on dashes
    and centered score display.
    """
    # Create the base visualization with exactly 30 dashes on each side
    left_side = "-" * 30
    right_side = "-" * 30
    
    # Insert the category markers after the third dash from zero on each side
    left_with_marker = left_side[:-3] + "|" + left_side[-3:]
    right_with_marker = right_side[:3] + "|" + right_side[3:]
    
    # Calculate dash position (-3.0 to +3.0 maps to 0 to 60 dashes)
    dash_position = int((score + 3.0) * 10)  # Each dash represents 0.1 units
    
    # Adjust position for markers and zero
    if score <= -0.3:
        position = dash_position
    elif score < 0:
        position = dash_position + 1
    elif score <= 0.3:
        position = dash_position + 2
    else:
        position = dash_position + 3
    
    # Clamp position to valid range
    position = max(0, min(63, position))  # 63 = 60 dashes + 2 markers + zero
    
    # Create score string with fixed width
    score_str = f"{score:.2f}"
    
    # Create the visualization
    visualization = "Low                        Moderate                        High\n"
    visualization += f"[{left_with_marker}0{right_with_marker}]\n"
    
    # Calculate the center position for the score
    score_offset = len(score_str) // 2
    padding = position - score_offset
    
    # Add pointer and score
    visualization += ' ' * position + "^\n"  # Pointer on its own line
    visualization += ' ' * padding + score_str  # Centered score below
    
    return visualization

def process_fasta_file(fasta_file, output_csv, model, scaler, variance_selector, feature_selector, feature_mask, all_feature_names, quiet=False):
    """Process FASTA file with sequences, classifying into three antigenicity categories."""
    results = []
    low_antigenicity_count = 0
    moderate_antigenicity_count = 0
    high_antigenicity_count = 0
    
    print(f"\nProcessing FASTA file: {fasta_file}")
    
    with open(fasta_file, 'r', encoding='utf-8') as fasta_handle, \
         open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Header', 'Sequence_Length', 'IAScore', 'Antigenicity_Category'])
        
        # Parse sequences manually to handle Unicode safely
        sequences = []
        current_header = ''
        current_sequence = ''
        
        for line in fasta_handle:
            line = line.strip()
            # Remove zero-width spaces and other problematic Unicode characters
            line = ''.join(c for c in line if c.isprintable() and ord(c) < 128)
            
            if line.startswith('>'):
                if current_header and current_sequence:
                    sequences.append((current_header, current_sequence))
                current_header = line[1:]  # Remove the '>' character
                current_sequence = ''
            elif line:  # Only add non-empty lines
                current_sequence += line
        
        # Don't forget the last sequence
        if current_header and current_sequence:
            sequences.append((current_header, current_sequence))
        
        total_sequences = len(sequences)
        print(f"Found {total_sequences} sequences")
        
        for i, (header, sequence) in enumerate(sequences, 1):
            header = header[:20]  # Truncate header to 20 characters
            
            try:
                # Clean the sequence
                cleaned_sequence = ''.join(c for c in sequence.upper() if c in 'ACDEFGHIKLMNPQRSTVWY')
                
                if not cleaned_sequence:
                    print(f"\nWarning: Invalid sequence for {header}")
                    csv_writer.writerow([header, 0, "Invalid sequence", "N/A"])
                    continue
                
                if len(cleaned_sequence) < 20:
                    print(f"\nWarning: Sequence too short for {header} ({len(cleaned_sequence)} aa)")
                    csv_writer.writerow([header, len(cleaned_sequence), "Sequence too short", "N/A"])
                    continue
                
                score = evaluate_sequence(cleaned_sequence, model, scaler, 
                                       variance_selector, feature_selector, 
                                       feature_mask, all_feature_names)
                                       
                inverted_score = -score
                
                # Determine antigenicity category
                if inverted_score < -0.3:
                    category = "Low"
                    low_antigenicity_count += 1
                elif inverted_score > 0.3:
                    category = "High"
                    high_antigenicity_count += 1
                else:
                    category = "Moderate"
                    moderate_antigenicity_count += 1
                
                if not quiet:
                    print(f"\nProcessing sequence {i}/{total_sequences}: {header}")
                    print(f"Intrinsic Antigenicity: {inverted_score:.2f} ({category})")
                    print(create_ascii_visualization(inverted_score))
                else:
                    print(f"\rProcessing: {i}/{total_sequences}", end='', flush=True)
                
                csv_writer.writerow([header, len(cleaned_sequence), f"{inverted_score:.2f}", category])
                results.append((header, inverted_score, category))
                
            except Exception as e:
                print(f"\nError processing sequence {header}: {str(e)}")
                csv_writer.writerow([header, len(sequence) if sequence else 0, "Error", "N/A"])
    
    if quiet:
        print()  # New line after progress indicator
        
    # Print summary
    print(f"\nAntigenicity Summary:")
    print(f"Low Antigenicity (score < -0.3): {low_antigenicity_count} sequences")
    print(f"Moderate Antigenicity (-0.3 to 0.3): {moderate_antigenicity_count} sequences")
    print(f"High Antigenicity (score > 0.3): {high_antigenicity_count} sequences")
    
    return results, low_antigenicity_count, moderate_antigenicity_count, high_antigenicity_count

def main():
    # Check and install dependencies
    check_and_install_dependencies()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Intrinsic Antigenicity Predictor")
    parser.add_argument("input_fasta", help="Input FASTA file")
    parser.add_argument("output_csv", nargs='?', help="Output CSV file (optional)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress detailed output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Force verbose output even for large files")
    args = parser.parse_args()

    input_fasta = args.input_fasta
    # Generate default output CSV filename if not provided
    if args.output_csv:
        output_csv = args.output_csv
    else:
        output_csv = os.path.splitext(input_fasta)[0] + '.csv'

    print(f"Input FASTA file: {input_fasta}")
    print(f"Output CSV file: {output_csv}")

    # Count the number of sequences in the input file
    with open(input_fasta, 'r') as f:
        num_sequences = sum(1 for line in f if line.startswith('>'))

    # Determine whether to quiet output
    if args.quiet:
        quiet = True
    elif args.verbose:
        quiet = False
    else:
        quiet = num_sequences > 25

    if quiet and not args.quiet:
        print(f"Note: Output is quiet by default for files with more than 25 sequences. Use -v or --verbose to override.")

    # Load models
    models_folder = "models"
    if not os.path.isdir(models_folder):
        print(f"Error: The '{models_folder}' folder does not exist in the current directory.")
        sys.exit(1)

    try:
        svm_model = load(os.path.join(models_folder, 'AntigenPred_SVM.joblib'))
        scaler = load(os.path.join(models_folder, 'AntigenPred_scaler.joblib'))
        variance_selector = load(os.path.join(models_folder, 'AntigenPred_variance_selector.joblib'))
        feature_selector = load(os.path.join(models_folder, 'AntigenPred_feature_selector.joblib'))
        feature_mask = load(os.path.join(models_folder, 'AntigenPred_feature_mask.joblib'))
        all_feature_names = load(os.path.join(models_folder, 'AntigenPred_all_feature_names.joblib'))
    except Exception as e:
        print(f"Error loading model files: {str(e)}")
        sys.exit(1)

    results, low_count, moderate_count, high_count = process_fasta_file(
        input_fasta, output_csv, svm_model, scaler, variance_selector, 
        feature_selector, feature_mask, all_feature_names, quiet
    )

    print(f"\nResults have been saved to {output_csv}")
if __name__ == "__main__":
    main()
