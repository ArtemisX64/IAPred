import subprocess
import argparse
import re
import os
import time
from datetime import datetime, timedelta

def format_time(seconds):
    return str(timedelta(seconds=round(seconds)))

def run_find_best_k():
    print("\nRunning Find_best_k.py...")
    start_time = time.time()
    
    process = subprocess.run(['python', 'Find_best_k.py'], 
                           capture_output=True, text=True)
    
    elapsed_time = time.time() - start_time
    print(f"Find_best_k.py completed in: {format_time(elapsed_time)}")
    
    output = process.stdout
    if process.returncode != 0:
        print("Error in Find_best_k.py:")
        print(process.stderr)
        return None
        
    k_match = re.search(r'Best number of features \(k\): (\d+)', output)
    if k_match:
        return int(k_match.group(1))
    return None

def run_optimize_c_gamma(k):
    print(f"\nRunning Optimize_C_and_gamma.py with k={k}...")
    start_time = time.time()
    
    process = subprocess.run(['python', 'Optimize_C_and_gamma.py', '--k', str(k)],
                           capture_output=True, text=True)
    
    elapsed_time = time.time() - start_time
    print(f"Optimize_C_and_gamma.py completed in: {format_time(elapsed_time)}")
    
    output = process.stdout
    if process.returncode != 0:
        print("Error in Optimize_C_and_gamma.py:")
        print(process.stderr)
        return None, None
        
    try:
        with open('c_gamma_optimization_results.txt', 'r') as f:
            content = f.read()
            c_match = re.search(r'Best C: ([\d.]+)', content)
            gamma_match = re.search(r'Best gamma: ([\d.e-]+)', content)
            
            if c_match and gamma_match:
                return float(c_match.group(1)), float(gamma_match.group(1))
    except:
        pass
        
    c_match = re.search(r'Best C: ([\d.]+)', output)
    gamma_match = re.search(r'Best gamma: ([\d.e-]+)', output)
    
    if c_match and gamma_match:
        return float(c_match.group(1)), float(gamma_match.group(1))
    return None, None

def run_remaining_scripts(k, c, gamma):
    scripts = [
        ('model_parameters.py', 'Model Parameters'),
        ('feature_importance.py', 'Feature Importance'),
        ('generate_and_save_models.py', 'Generate and Save Models')
    ]
    
    total_start_time = time.time()
    for script, name in scripts:
        print(f"\nRunning {name}...")
        start_time = time.time()
        
        process = subprocess.run([
            'python', script,
            '--k', str(k),
            '--c', str(c),
            '--gamma', str(gamma)
        ], capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        print(f"{script} completed in: {format_time(elapsed_time)}")
        
        if process.returncode != 0:
            print(f"Error in {script}:")
            print(process.stderr)
            return False
        print(process.stdout)
    
    total_elapsed = time.time() - total_start_time
    print(f"\nAll training scripts completed in: {format_time(total_elapsed)}")
    return True

def run_cross_validations(k, c, gamma):
    cv_scripts = [
        ('10fold_CV.py', '10-fold Cross Validation'),
        ('LOCO-CV.py', 'Leave-One-Class-Out Cross Validation'),
        ('LOPO-CV.py', 'Leave-One-Pathogen-Out Cross Validation')
    ]
    
    total_start_time = time.time()
    for script, name in cv_scripts:
        print(f"\nRunning {name}...")
        start_time = time.time()
        
        process = subprocess.run([
            'python', script,
            '--k', str(k),
            '--c', str(c),
            '--gamma', str(gamma)
        ], capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        print(f"{script} completed in: {format_time(elapsed_time)}")
        
        if process.returncode != 0:
            print(f"Error in {script}:")
            print(process.stderr)
            return False
        print(process.stdout)
    
    total_elapsed = time.time() - total_start_time
    print(f"\nAll cross-validations completed in: {format_time(total_elapsed)}")
    return True

def run_internal_evaluation():
    print("\nRunning Internal Evaluation...")
    start_time = time.time()
    
    process = subprocess.run(['python', 'Internal_Evaluation.py'],
                           capture_output=True, text=True)
    
    elapsed_time = time.time() - start_time
    print(f"Internal evaluation completed in: {format_time(elapsed_time)}")
    
    if process.returncode != 0:
        print("Error in Internal_Evaluation.py:")
        print(process.stderr)
        return False
    print(process.stdout)
    return True

def run_external_evaluation():
    print("\nRunning External Evaluation...")
    start_time = time.time()
    
    process = subprocess.run(['python', 'External_Evaluation.py'],
                           capture_output=True, text=True)
    
    elapsed_time = time.time() - start_time
    print(f"External evaluation completed in: {format_time(elapsed_time)}")
    
    if process.returncode != 0:
        print("Error in External_Evaluation.py:")
        print(process.stderr)
        return False
    print(process.stdout)
    return True

def main():
    total_start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Run training pipeline')
    parser.add_argument('--k', type=int, help='Number of features')
    parser.add_argument('--c', type=float, help='SVM C parameter')
    parser.add_argument('--gamma', type=float, help='SVM gamma parameter')
    args = parser.parse_args()

    k = args.k
    c = args.c
    gamma = args.gamma

    if k is None:
        k = run_find_best_k()
        if k is None:
            print("Failed to find best k")
            return

    if c is None or gamma is None:
        c, gamma = run_optimize_c_gamma(k)
        if c is None or gamma is None:
            print("Failed to optimize C and gamma")
            return

    print(f"\nUsing parameters: k={k}, C={c}, gamma={gamma}")
    
    if not run_remaining_scripts(k, c, gamma):
        print("Training pipeline failed")
        return

    if not run_cross_validations(k, c, gamma):
        print("Cross-validation evaluations failed")
        return

    # Run external evaluation before internal evaluation
    if not run_external_evaluation():
        print("External evaluation failed")
        return

    if not run_internal_evaluation():
        print("Internal evaluation failed")
        return

    total_elapsed = time.time() - total_start_time
    print(f"\nPipeline completed successfully!")
    print(f"Total execution time: {format_time(total_elapsed)}")

if __name__ == "__main__":
    main()
