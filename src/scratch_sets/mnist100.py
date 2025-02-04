import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import os
from time import time
from grid_search import grid_search

# https://www.kaggle.com/datasets/martininf1n1ty/mnist100/data
data = np.load(os.path.join("data", "mnist100_compressed.npz"))
X_test, Y_test, X_train, Y_train = (
    data["test_images"],
    data["test_labels"],
    data["train_images"],
    data["train_labels"],
)
# Print shapes with labels
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Data booleanization
X_train = np.where(X_train > 75, 1, 0)
X_test = np.where(X_test > 75, 1, 0)

X_train = X_train.reshape(X_train.shape[0], 28 * 56)
X_test = X_test.reshape(X_test.shape[0], 28 * 56)

# Grid search
num_clauses_values = [500, 1000, 2000]
threshold_values = [10, 70, 100, 250, 200, ]
specificity_values = [2.0, 5.0, 10.0, 17.0]
epochs = 5

best_params = grid_search(
    X_train,
    Y_train,
    X_test,
    Y_test,
    num_clauses_values=num_clauses_values,
    threshold_values=threshold_values,
    specificity_values=specificity_values,
    epochs=epochs,
)
print(best_params)
