from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np 
import os
from matrepr import mprint
from pickle import load, dump
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#tm = MultiClassTsetlinMachine(500, 15, 3.9, boost_true_positive_feedback=0)
#tm.fit(X_train, Y_train, epochs=200)
#dump(tm, open("tm.pkl", "wb"))
#
tm = load(open("tm.pkl", "rb"))

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

r1 = tm.predict(np.array([[1,0,1,0,1,0,1,1,1,1,0,0]]))
r2 = tm.predict(np.array([[0,1,1,0,1,0,1,1,1,1,0,0]]))
r3 = tm.predict(np.array([[0,0,1,0,1,0,1,1,1,1,0,0]]))
r4 = tm.predict(np.array([[1,1,1,0,1,0,1,1,1,1,0,0]]))

print(f"Prediction: x1 = 1, x2 = 0, ... -> y = {r1}")
print(f"Prediction: x1 = 0, x2 = 1, ... -> y = {r2}")
print(f"Prediction: x1 = 0, x2 = 0, ... -> y = {r3}")
print(f"Prediction: x1 = 1, x2 = 1, ... -> y = {r4}")

transformed = tm.transform(X_train)
print(transformed.shape)

X = transformed[0:1]
mprint(X)

