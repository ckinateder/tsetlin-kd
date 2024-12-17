# -*- coding: utf-8 -*-
"""TM Knowledge Distilliation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1itcZohE6OlXICa9NXAiXn7w1tMeU4paY
"""
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from tensorflow.keras.datasets import mnist, cifar10
import numpy as np
from time import time

def run_mnist(
    teacher_num_clauses = 400,
    teacher_T = 10,
    teacher_s = 5,
    teacher_epochs = 60,
    student_num_clauses = 100,
    student_T = 10,
    student_s = 5,
    student_epochs = 60
):
    print("Running MNIST")
    """### Load MNIST data"""
    (X_train, Y_train), (X_val, Y_val) = mnist.load_data()

    # Data booleanization
    X_train = np.where(X_train > 75, 1, 0)
    X_val = np.where(X_val > 75, 1, 0)

    # Input data flattening
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_val = X_val.reshape(X_val.shape[0], 28*28)
    Y_train = Y_train.flatten()
    Y_val = Y_val.flatten()

    """### Set params for the models"""

    teacher_num_clauses = 400
    teacher_T = 10
    teacher_s = 5
    teacher_epochs = 60

    student_num_clauses = 100
    student_T = teacher_T
    student_s = teacher_s
    student_epochs = 60

    """### Train the teacher model"""

    #create Teacher
    teacher_tm = MultiClassTsetlinMachine(teacher_num_clauses, teacher_T, teacher_s, number_of_state_bits=8)
    #train Teacher on the original dataset
    start = time()
    teacher_tm.fit(X_train, Y_train, epochs=teacher_epochs)
    end = time()
    print(f'Teacher training time: {end-start:.2f} s')

    #evaluate Teacher training and validation accuracy
    acc_train_1 = 100*(teacher_tm.predict(X_train) == Y_train).mean()
    acc_val_1 = 100*(teacher_tm.predict(X_val) == Y_val).mean()
    print(f'Teacher training accuracy ({teacher_epochs} training epochs):   {acc_train_1:.2f}%')
    print(f'Teacher validation accuracy ({teacher_epochs} training epochs): {acc_val_1:.2f}%')

    """### Train the student model from the teacher model"""

    #create Student
    student_tm = MultiClassTsetlinMachine(student_num_clauses, student_T, student_s, number_of_state_bits=8)
    #tune Student using as an input Teacher clause outputs
    #tm.transform method returns raw TM clause outputs corresponding to the input dataset
    start = time()
    student_tm.fit(teacher_tm.transform(X_train), Y_train, epochs=student_epochs)
    end = time()
    print(f'Student training time: {end-start:.2f} s')

    #evaluate Student training and validation accuracy; notice how is the input defined
    acc_train_2 = 100*(student_tm.predict(teacher_tm.transform(X_train)) == Y_train).mean()
    acc_val_2 = 100*(student_tm.predict(teacher_tm.transform(X_val)) == Y_val).mean()
    print(f'Student training accuracy (100 tuning epochs):   {acc_train_2:.2f}%')
    print(f'Student validation accuracy (100 tuning epochs): {acc_val_2:.2f}%')

    """### View Accuracy"""

    #evaluate Teacher training and validation accuracy
    print(f'Teacher training accuracy ({teacher_epochs} epochs):   {acc_train_1:.2f}%')
    print(f'Student training accuracy ({student_epochs} epochs):   {acc_train_2:.2f}%')
    print(f'Teacher validation accuracy ({teacher_epochs} epochs): {acc_val_1:.2f}%')
    print(f'Student validation accuracy ({student_epochs} epochs): {acc_val_2:.2f}%')

if __name__ == '__main__':
    run_mnist()