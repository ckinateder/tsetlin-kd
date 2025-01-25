from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np 
import os

train_data = np.loadtxt(os.path.join("data", "2DNoisyXORTrainData.txt"))
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt(os.path.join("data", "2DNoisyXORTestData.txt"))
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)

tm.fit(X_train, Y_train, epochs=200)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

r1 = tm.predict(np.array([[1,0,1,0,1,0,1,1,1,1,0,0]]))
r2 = tm.predict(np.array([[0,1,1,0,1,0,1,1,1,1,0,0]]))
r3 = tm.predict(np.array([[0,0,1,0,1,0,1,1,1,1,0,0]]))
r4 = tm.predict(np.array([[1,1,1,0,1,0,1,1,1,1,0,0]]))

print(f"Prediction: x1 = 1, x2 = 0, ... -> y = {r1}")
print(f"Prediction: x1 = 0, x2 = 1, ... -> y = {r2}")
print(f"Prediction: x1 = 0, x2 = 0, ... -> y = {r3}")
print(f"Prediction: x1 = 1, x2 = 1, ... -> y = {r4}")


tm2 = MultiClassTsetlinMachine(50, 15, 3.9, boost_true_positive_feedback=0)
tm2.fit(tm.transform(X_train), Y_train, epochs=50)
import pdb
pdb.set_trace()