from torchvision.datasets import KMNIST
from torchvision import transforms
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time

train = KMNIST(root="data", download=True, train=True, transform=transforms.ToTensor())
test = KMNIST(root="data", download=True, train=False, transform=transforms.ToTensor())

X_train, Y_train = train.data.numpy(), train.targets.numpy()
X_test, Y_test = test.data.numpy(), test.targets.numpy()

X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)

X_train = np.where(X_train > 75, 1, 0)
X_test = np.where(X_test > 75, 1, 0)

tm = MultiClassTsetlinMachine(100, 500, 10.0, weighted_clauses=True)
for i in range(30):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result = 100*(tm.predict(X_test) == Y_test).mean()
    stop_testing = time()
    print(f'Epoch {i:>3}: Training time: {stop_training-start_training:.2f} s, Testing time: {stop_testing-start_testing:.2f} s, Test accuracy: {result:.2f}%')

print("--\nAccuracy:", 100*(tm.predict(X_test) == Y_test).mean())