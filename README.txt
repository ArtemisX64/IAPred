# Intrinsic Antigenicity predictor

## Overview

This software predicts the Intrinsic Antigenicity of infectious disease pathogens proteins, but in a pathogen-and-host independant way. The predictor was trained using known antigens from virus, bacteria (gram+ and gram-), fungi, protozoos and helminth, in infectious diseases affecting human, mice, pigs, dogs, goats, rabbit, guinea pigs, cattle, ewe and sheep

## Features

- Processes protein sequences in FASTA format
- Provides an intrinsic antigenicity score for each protein
- Generates a visual ASCII representation of the antigenicity score
- Outputs results in CSV format for further analysis

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - numpy
  - biopython
  - scikit-learn
  - joblib

## Installation

    pip install -r requirements.txt

    or

    pip install numpy biopython scikit-learn joblib


   Note: The script will attempt to install these packages automatically if they are not found, but might need permisions and need to be done manually.


## Usage

To run IntrinsicAntigeniciy, use the following command:

```
python IntrinsicAntigenicity.py input_fasta_file [output_csv_file]
```

- `input_fasta_file`: Path to the input FASTA file containing protein sequences (required)
- `output_csv_file`: Path to the output CSV file (optional). If not provided, the output will be saved with the same name as the input file but with a .csv extension.

Example usage:
```
python IntrinsicAntigeniciy.py H_pylori.fasta
```
or
```
python IntrinsicAntigeniciy.py H_pylori.fasta H_pylori_results.csv
```

## Output

The tool provides two types of output:

1. Console output: For each protein sequence, it displays:
   - The protein header (first 20 characters)
   - The intrinsic antigenicity score
   - An ASCII visualization of the score in a antigenicity scale

2. CSV file: Contains two columns:
   - Header: The protein header (first 20 characters)
   - Intrinsic Antigenicity Score: The calculated score for each protein

## Interpreting Results

- The intrinsic antigenicity score typically ranges from -3 to 3, although there are not theoretical max or min scores.
- Higher positive scores indicate higher predicted antigenicity.
- Lower negative scores indicate lower predicted antigenicity.
- This predictor only takes into account the amino acid sequence, but the real antigenicity might be influence by post-translational modification, precense of T- and B- cell epitopes, and the disponibility of the antigen in an infection/immunization. This predictor is intended to be use only as a broad classification tool

## Troubleshooting

If you encounter any issues:
1. Ensure all required packages are installed correctly.
2. Check that the "models" folder with all necessary .joblib files is present in the script directory (6 files).
3. Check that the ¨function.py" and "protein_motifs.txt" files are present in the script directoty
3. Verify that your input FASTA file is formatted correctly.

For further assistance, please contact [Your Contact Information].

## License

[Include your license information here]

## Citation

If you use IntrinsicAntigeniciy in your research, please cite:
[Include citation information here]
