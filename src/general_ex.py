# -*- coding: utf-8 -*-
"""TM Knowledge Distilliation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1itcZohE6OlXICa9NXAiXn7w1tMeU4paY
"""
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
from time import time

# set seeds
np.random.seed(0)


def run_general_experiment(
    X_train, Y_train, X_val, Y_val,
    experiment_name,
    params = {
        "teacher_num_clauses": 400,
        "student_num_clauses": 200,
        "T": 10,
        "s": 5,
        "teacher_epochs": 60,
        "student_epochs": 60
    },
):
    print(f"Running {experiment_name}")
    print(params)
    assert len(X_train) == len(Y_train)
    assert len(X_val) == len(Y_val)

    assert "teacher_num_clauses" in params
    assert "T" in params
    assert "s" in params
    assert "teacher_epochs" in params
    assert "student_num_clauses" in params
    assert "student_epochs" in params
    assert params["teacher_num_clauses"] > params["student_num_clauses"], "Student clauses should be less than teacher clauses"

    teacher_num_clauses = params["teacher_num_clauses"]
    T = params["T"]
    s = params["s"]
    teacher_epochs = params["teacher_epochs"]
    student_num_clauses = params["student_num_clauses"]
    student_epochs = params["student_epochs"]
    combined_epochs = teacher_epochs + student_epochs


    """### Train the baseline student model"""
    # create student but don't train on teacher's output
    print(f"Creating a baseline student with {student_num_clauses} clauses and training on original data")
    baseline_tm = MultiClassTsetlinMachine(student_num_clauses, T, s, number_of_state_bits=8)
    start = time()
    baseline_tm.fit(X_train, Y_train, epochs=combined_epochs)
    end = time()
    print(f'Baseline training time: {end-start:.2f} s')

    #evaluate Teacher training and validation accuracy
    acc_train_baseline = 100*(baseline_tm.predict(X_train) == Y_train).mean()
    acc_val_baseline = 100*(baseline_tm.predict(X_val) == Y_val).mean()
    print(f'Baseline training accuracy ({combined_epochs} training epochs):   {acc_train_baseline:.2f}%')
    print(f'Baseline validation accuracy ({combined_epochs} training epochs): {acc_val_baseline:.2f}%')

    """### Train the teacher model"""

    #create Teacher
    print(f"Creating teacher with {teacher_num_clauses} clauses")
    teacher_tm = MultiClassTsetlinMachine(teacher_num_clauses, T, s, number_of_state_bits=8)
    #train Teacher on the original dataset
    start = time()
    teacher_tm.fit(X_train, Y_train, epochs=teacher_epochs)
    end = time()
    print(f'Teacher training time: {end-start:.2f} s')

    #evaluate Teacher training and validation accuracy
    acc_train_teacher = 100*(teacher_tm.predict(X_train) == Y_train).mean()
    acc_val_teacher = 100*(teacher_tm.predict(X_val) == Y_val).mean()
    print(f'Teacher training accuracy ({teacher_epochs} training epochs):   {acc_train_teacher:.2f}%')
    print(f'Teacher validation accuracy ({teacher_epochs} training epochs): {acc_val_teacher:.2f}%')

    """### Train the student model from the teacher model"""

    #create Student
    print(f"Creating student with {student_num_clauses} clauses and training on teacher's output (teacher has trained {teacher_epochs} epochs)")
    student_tm = MultiClassTsetlinMachine(student_num_clauses, T, s, number_of_state_bits=8)
    start = time()
    student_tm.fit(teacher_tm.transform(X_train), Y_train, epochs=student_epochs)
    end = time()
    print(f'Student training time: {end-start:.2f} s')

    #evaluate Student training and validation accuracy; notice how is the input defined
    acc_train_student = 100*(student_tm.predict(teacher_tm.transform(X_train)) == Y_train).mean()
    acc_val_student = 100*(student_tm.predict(teacher_tm.transform(X_val)) == Y_val).mean()
    print(f'Student training accuracy ({student_epochs} training epochs):   {acc_train_student:.2f}%')
    print(f'Student validation accuracy ({student_epochs} training epochs): {acc_val_student:.2f}%')

    """### Train the teacher model more with the student epochs"""
    print(f"Training teacher for {student_epochs} more epochs (total {combined_epochs})")
    start = time()
    teacher_tm.fit(X_train, Y_train, epochs=student_epochs)
    end = time()
    print(f'Teacher training time: {end-start:.2f} s')

    """### View Accuracy"""
    results = {
        "params": params,
        "experiment_name": experiment_name,
        "acc_train_baseline": acc_train_baseline,
        "acc_val_baseline": acc_val_baseline,
        "acc_train_teacher": acc_train_teacher,
        "acc_val_teacher": acc_val_teacher,
        "acc_train_student": acc_train_student,
        "acc_val_student": acc_val_student
    }

    #evaluate Teacher training and validation accuracy
    print(f"\nResults for {experiment_name}:")
    print(f"Teacher num clauses: {teacher_num_clauses}, student and baseline num clauses: {student_num_clauses}, T: {T}, s: {s}")
    print(f"Student is trained on teacher's output, baseline is trained on original data")
    print("Training accuracy:")
    print(f'- Baseline training accuracy ({combined_epochs} epochs):   {acc_train_baseline:.2f}%')
    print(f'- Teacher training accuracy ({teacher_epochs}+{student_epochs} epochs):    {acc_train_teacher:.2f}%')
    print(f'- Student training accuracy ({teacher_epochs}+{student_epochs} epochs):    {acc_train_student:.2f}%')
    print("Validation accuracy:")
    print(f'- Baseline validation accuracy ({combined_epochs} epochs): {acc_val_baseline:.2f}%')
    print(f'- Teacher validation accuracy ({teacher_epochs}+{student_epochs} epochs):  {acc_val_teacher:.2f}%')
    print(f'- Student validation accuracy ({teacher_epochs}+{student_epochs} epochs):  {acc_val_student:.2f}%')


    # compute the difference in accuracy and parameter count
    acc_diff = acc_val_student - acc_val_baseline
    print(f'Accuracy difference (student-baseline): {acc_diff:.2f} pts')
    # parameter reduction ratio
    param_ratio = student_num_clauses / teacher_num_clauses
    print(f'Parameter ratio (student:teacher): {param_ratio:.2f}:1')


if __name__ == '__main__':
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

    run_general_experiment(X_train, Y_train, X_val, Y_val, "MNIST")

    """### Load Fashion MNIST data"""
    (X_train, Y_train), (X_val, Y_val) = fashion_mnist.load_data()
    # Data booleanization
    X_train = np.where(X_train > 75, 1, 0)
    X_val = np.where(X_val > 75, 1, 0)
    
    # Input data flattening
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_val = X_val.reshape(X_val.shape[0], 28*28)
    Y_train = Y_train.flatten()
    Y_val = Y_val.flatten()

    run_general_experiment(X_train, Y_train, X_val, Y_val, "Fashion MNIST")
