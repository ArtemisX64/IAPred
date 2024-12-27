# IApred Training Pipeline 🧬

Training pipeline for IApred (Intrinsic Antigenicity Predictor).

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/username/IApred/graphs/commit-activity)

## 📋 Overview

This repository contains the training pipeline for IApred's SVM-based model. All scripts can run independently or together via `run_training.py`.

## 🚀 Usage Options

### Pipeline Mode
Run complete pipeline with automatic parameter optimization:
```bash
python run_training.py
```

### Independent Mode
Run scripts individually with parameters. In fo parameters are assigned, each script will run with the default values:
- `k = 529` (number of features)
- `c = 1` (SVM C parameter) 
- `gamma = 0.01` (SVM gamma parameter)

```bash
python model_parameters.py --k 529 --c 1 --gamma 0.01
```

## 📊 Pipeline Components

### Core Scripts

| Script | Purpose | Usage |
|--------|---------|--------|
| `run_training.py` | Main pipeline orchestrator | `python run_training.py` |
| `Find_best_k.py` | Optimal number of feature selection | `python Find_best_k.py` |
| `Optimize_C_and_gamma.py` | SVM parameter optimization | `python Optimize_C_and_gamma.py --k 529` |
| `model_parameters.py` | Model training and evaluation | `python model_parameters.py --k 529 --c 1 --gamma 0.01` |

### Evaluation Scripts

| Script | Purpose | Usage |
|--------|---------|--------|
| `10fold_CV.py` | Standard cross-validation | `python 10fold_CV.py --k 529 --c 1 --gamma 0.01` |
| `LOCO-CV.py` | Leave-One-Class-Out validation | `python LOCO-CV.py --k 529 --c 1 --gamma 0.01` |
| `LOPO-CV.py` | Leave-One-Pathogen-Out validation | `python LOPO-CV.py --k 529 --c 1 --gamma 0.01` |
| `Internal_Evaluation.py` | Internal dataset evaluation | `python Internal_Evaluation.py` |
| `External_Evaluation.py` | External dataset comparison | `python External_Evaluation.py` |

### Support Scripts
- `feature_importance.py`: Feature analysis and visualization
- `generate_and_save_models.py`: Model persistence
- `functions_for_training.py`: Utility functions

## 📂 Output

Results are saved in `TrainingResults/`

Models are saved in `Models/`


## 📝 Requirements

- Python 3.6+
- Dependencies:
  ```bash
  pip install numpy scikit-learn imbalanced-learn matplotlib seaborn biopython joblib pandas scipy
  ```

## 📫 Contact

For support or queries, please [open an issue](https://github.com/sebamiles/IApred/issues) or contact [smiles@higiene.edu.uy].

## 🤝 Contributing

Contributions welcome! Please submit a Pull Request.

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.
