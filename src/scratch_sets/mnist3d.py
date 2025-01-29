import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import os
from time import time
from grid_search import grid_search

# https://www.kaggle.com/datasets/daavoo/3d-mnist
# load with h5py
import h5py

with h5py.File(os.path.join("data", "mnist3d.h5"), "r") as hf:
    X_train = hf["X_train"][:]
    Y_train = hf["y_train"][:]    
    X_test = hf["X_test"][:]  
    Y_test = hf["y_test"][:]  

# Print shapes with labels
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Data booleanization
X_train = np.where(X_train > 0.3, 1, 0)
X_test = np.where(X_test > 0.3, 1, 0)

X_train = X_train.reshape(X_train.shape[0], 16*16*16)
X_test = X_test.reshape(X_test.shape[0], 16*16*16)

best_params = grid_search(X_train, Y_train, X_test, Y_test, num_clauses_values=[500, 1000, 4000], threshold_values=[50, 100, 150, 200], specificity_values=[10.0, 15.0, 20.0, 30.0], epochs=5)

print(best_params)