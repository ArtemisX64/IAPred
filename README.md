# IApred: Intrinsic Antigenicity Predictor 🧬

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/username/IApred/graphs/commit-activity)

IApred is a powerful tool for predicting the intrinsic antigenicity of pathogen proteins in a host-independent manner. Our predictor leverages a manually curated dataset spanning multiple pathogen types and host species to provide accurate antigenicity predictions.

## 🚀 Quick Start

### Option 1: Google Colab (No Installation Required)
Try IApred instantly in your browser:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](INSERT_COLAB_LINK_HERE)

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://github.com/username/IApred.git
cd IApred

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Features

- 🔍 Accurate prediction of protein antigenicity
- 📊 Visual representation of antigenicity scores
- 📝 Detailed CSV output for further analysis
- 🧮 Support for both single sequences and batch processing
- 🎨 ASCII and color visualization options (only on google colab)

## 📋 Requirements

- Python 3.6+
- Dependencies:
  - numpy
  - biopython
  - scikit-learn
  - joblib
  - scipy

## 💻 Usage

### Command Line Interface
```bash
python IApred.py input_fasta_file [output_csv_file]
```

### Example
```bash
python IApred.py test.fasta
# or
python IApred.py test.fasta test_results.csv
```

## 📊 Output Format

### Console Output
```
Processing sequence: >Protein_X
Intrinsic Antigenicity: 0.63 (High)
Low                        Moderate                        High
[---------------------------|---0---|---------------------------]
                                       ^
                                     0.63
```

### CSV Output
| Header | Sequence_Length | IAscore | Antigenicity_Category |
|--------|----------------|---------|---------------------|
| Protein_X | 245 | 1.25 | High |

## 🎯 Interpreting Results

- Score Range: Typically -3 to 3
- Categories:
  - High: > 0.3
  - Moderate: -0.3 to 0.3
  - Low: < -0.3

> **Note**: The predictor focuses on amino acid sequence-based antigenicity. Actual antigenicity may be influenced by additional factors such as structure, post-translational modifications and epitope availability.

## ❗ Troubleshooting

1. Verify installation of required packages
2. Ensure presence of:
   - `models` folder with all .joblib files
   - `functions.py`
   - `protein_motifs.txt`
3. Check FASTA file formatting

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use IApred in your research, please cite:
```bibtex
[Citation information will be added upon publication]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📫 Contact

For support or queries, please [open an issue](https://github.com/sebamiles/IApred/issues) or contact [smiles@higiene.edu.uy].
