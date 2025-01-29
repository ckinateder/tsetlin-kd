import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from tensorflow.keras.datasets import mnist
from time import time
from tqdm import tqdm, trange

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

    progress_bar = trange(total_iterations, desc="Grid Search Progress")
    current_iter = 0

    # Grid search
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
                        f"Grid Search Progress (Best: {best_accuracy:.2f}%)"
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
    # Load and prepare data with mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
    X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

    best_params = grid_search(
        X_train,
        Y_train,
        X_test,
        Y_test,
        num_clauses_values=[100, 400, 1000],
        threshold_values=[10, 50, 150],
        specificity_values=[3.0, 10.0, 15.0],
        epochs=5,
    )
