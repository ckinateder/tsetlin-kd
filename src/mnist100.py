import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import os
from time import time

# https://www.kaggle.com/datasets/martininf1n1ty/mnist100/data
data = np.load(os.path.join("data", "mnist100_compressed.npz"))
X_test, Y_test, X_train, Y_train =  data['test_images'], data['test_labels'], data['train_images'], data['train_labels']
# Print shapes with labels
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Data booleanization
X_train = np.where(X_train > 75, 1, 0)
X_test = np.where(X_test > 75, 1, 0)

X_train = X_train.reshape(X_train.shape[0], 28*56)
X_test = X_test.reshape(X_test.shape[0], 28*56)

tm = MultiClassTsetlinMachine(4000, 200, 20.0)
epochs = 30
print(f"\nAccuracy over {epochs} epochs:\n")
for i in range(epochs):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

print(tm.predict(X_test))