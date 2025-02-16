import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from tensorflow.keras.datasets import mnist, fashion_mnist
from time import time
from tqdm import tqdm, trange
import h5py
import os
from datasets import prepare_imdb_data, IMDBDataset
from util import load_or_create

def grid_search(
    X_train,
    Y_train,
    X_test,
    Y_test,
    num_clauses_values=[2000, 4000, 6000],
    threshold_values=[100, 150, 200],
    specificity_values=[10.0, 15.0, 20.0],
    other_params={},
    epochs=5,
    random_search=False,
):
    """
    Perform grid search to find optimal Tsetlin Machine parameters.

    Args:
        X_train: Training data features
        Y_train: Training data labels
        X_test: Test data features
        Y_test: Test data labels
        num_clauses_values: List of number of clauses to try
        threshold_values: List of threshold values to try
        specificity_values: List of specificity values to try
        epochs: Number of epochs to train each configuration

    Returns:
        dict: Best parameters found and their accuracy
    """
    best_accuracy = 0
    best_params = {}
    
    total_iterations = len(num_clauses_values) * len(threshold_values) * len(specificity_values) * epochs
    tqdm.write(f"Performing grid search over\n- num_clauses: {num_clauses_values}\n- threshold: {threshold_values}\n- specificity: {specificity_values}\n- epochs: {epochs}")
    tqdm.write(f"- other parameters: {other_params}")
    tqdm.write(f"Total iterations: {total_iterations}")

    progress_bar = trange(total_iterations, desc="Grid Search")
    current_iter = 0

    # Grid search
    if random_search:
        num_clauses_values = np.random.choice(num_clauses_values, size=len(num_clauses_values), replace=False)
        threshold_values = np.random.choice(threshold_values, size=len(threshold_values), replace=False)
        specificity_values = np.random.choice(specificity_values, size=len(specificity_values), replace=False)

    for num_clauses in num_clauses_values:
        for threshold in threshold_values:
            for specificity in specificity_values:
                tqdm.write(
                    f"\nTesting parameters: clauses={num_clauses}, threshold={threshold}, s={specificity}"
                )

                # Create Tsetlin Machine with given parameters
                tm = MultiClassTsetlinMachine(
                    num_clauses, threshold, specificity, **other_params
                )

                # Train for specified number of epochs to get a reasonable estimate
                for i in range(epochs):
                    start_training = time()
                    tm.fit(X_train, Y_train, epochs=1, incremental=True)
                    stop_training = time()

                    start_testing = time()
                    result = 100 * (tm.predict(X_test) == Y_test).mean()
                    stop_testing = time()

                    tqdm.write(
                        "#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs"
                        % (
                            i + 1,
                            result,
                            stop_training - start_training,
                            stop_testing - start_testing,
                        )
                    )

                    # Update best parameters if we found better accuracy
                    if result > best_accuracy:
                        best_accuracy = result
                        best_params = {
                            "num_clauses": num_clauses,
                            "threshold": threshold,
                            "specificity": specificity,
                            "epoch": i + 1,
                            "accuracy": result,
                        }
                    
                    current_iter += 1
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Grid Search (Best: {best_accuracy:.2f}% [{best_params['num_clauses']}, {best_params['threshold']}, {best_params['specificity']}])"
                    )

    progress_bar.close()
    print("\nBest parameters found:")
    print(f"Number of clauses: {best_params['num_clauses']}")
    print(f"Threshold: {best_params['threshold']}")
    print(f"Specificity: {best_params['specificity']}")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Found at epoch: {best_params['epoch']}")

    return best_params


if __name__ == "__main__":
    # do IMDB grid search
    imdb_dataset = load_or_create(os.path.join("data", "imdb_dataset.pkl"), IMDBDataset)

    X_train, Y_train, X_test, Y_test = imdb_dataset.get_data()

    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

    best_params = grid_search(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_clauses_values=[2000],
        threshold_values=[500, 2000, 4000, 6000, 8000, 10000],
        specificity_values=[3, 5.0,10.0, 20.0, 40.0],
        epochs=5,
    )

    # Load and prepare data with mnist-3d
    """### Load MNIST-3D data"""
    with h5py.File(os.path.join("data", "mnist3d.h5"), "r") as hf:
        X_train = hf["X_train"][:]
        Y_train = hf["y_train"][:]    
        X_test = hf["X_test"][:]  
        Y_test = hf["y_test"][:]  

    # Print shapes with labels
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

    # Data booleanization
    X_train = np.where(X_train > 0.32, 1, 0)
    X_test = np.where(X_test > 0.32, 1, 0)

    best_params = grid_search(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_clauses_values=[150, 1500],
        threshold_values=[20, 45, 60, 100],
        specificity_values=[3.0, 8.0, 15.0, 30.0, 50.0],
        epochs=5,
    )

    print(best_params)
    # Load and prepare data with fashion mnist
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
    X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

    best_params = grid_search(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_clauses_values=[750],
        threshold_values=[5, 8, 10, 30, 40, 50],
        specificity_values=[10.0, 15.0, 25.0, 40.0],
        epochs=5,
        random_search=True,
    )

    print(best_params)