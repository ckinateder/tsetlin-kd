import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from time import time
from tqdm import tqdm, trange
import h5py
import os
from datasets import prepare_imdb_data, IMDBDataset, MNISTDataset, FashionMNISTDataset, KMNISTDataset
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

                    start_transform = time()
                    tm.transform(X_test)
                    stop_transform = time()

                    tqdm.write(
                        f"#{i + 1} Accuracy: {result:.2f}% Training: {stop_training - start_training:.2f}s Testing: {stop_testing - start_testing:.2f}s Transform:{stop_transform - start_transform:.2f}s"
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
    # do KMNIST grid search
    kmnist_dataset = load_or_create(os.path.join("data", "kmnist_dataset.pkl"), KMNISTDataset)

    X_train, Y_train, X_test, Y_test = kmnist_dataset.get_data()

    grid_search(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_clauses_values=[500],
        threshold_values=[40, 100, 300, 500, 2000],
        specificity_values=[5, 9.5, 20.0, 40.0],
        epochs=5,
        random_search=True
    )