
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import numpy as np
from time import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

# set seeds
np.random.seed(0)


def run_detailed_experiment(
    X_train, Y_train, X_test, Y_test,
    experiment_name,
    params={
        "teacher_num_clauses": 400,
        "student_num_clauses": 200,
        "T": 10,
        "s": 5,
        "teacher_epochs": 60,
        "student_epochs": 60
    },
) -> pd.DataFrame:
    """
    Train a baseline student model with teacher_epochs + student_epochs epochs
    Train a baseline teacher model with teacher_epochs + student_epochs epochs
    Rebuild the teacher model and train it with teacher_epochs epochs
    Train the student model with student_epochs epochs on the teacher's output

    """

    print(f"Running {experiment_name}")
    print(params)
    assert len(X_train) == len(Y_train)
    assert len(X_test) == len(Y_test)

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

    # create a results dataframe
    results = pd.DataFrame(columns=["acc_test_teacher", "acc_test_student", "acc_test_distilled", "time_train_teacher", "time_train_student",
                           "time_train_distilled", "time_test_teacher", "time_test_student", "time_test_distilled"], index=range(combined_epochs))

    """### Train the baseline student model"""
    print(
        f"Creating a baseline student with {student_num_clauses} clauses and training on original data")
    baseline_distilled_tm = MultiClassTsetlinMachine(
        student_num_clauses, T, s, number_of_state_bits=8)

    start = time()

    # train baseline
    for i in range(combined_epochs):
        start_training = time()
        baseline_distilled_tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result = 100*(baseline_distilled_tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        results.loc[i, "acc_test_student"] = result
        results.loc[i, "time_train_student"] = stop_training-start_training
        results.loc[i, "time_test_student"] = stop_testing-start_testing

        print(f'Epoch {i:>3}: Training time: {stop_training-start_training:.2f} s, Testing time: {stop_testing-start_testing:.2f} s, Test accuracy: {result:.2f}%')

    end = time()

    print(f'Baseline student training time: {end-start:.2f} s')

    """### Train the baseline teacher model"""
    print(
        f"Creating a baseline teacher with {student_num_clauses} clauses and training on original data")
    baseline_teacher_tm = MultiClassTsetlinMachine(
        teacher_num_clauses, T, s, number_of_state_bits=8)

    start = time()

    # train baseline
    for i in range(combined_epochs):
        start_training = time()
        baseline_teacher_tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result = 100*(baseline_teacher_tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        results.loc[i, "acc_test_teacher"] = result
        results.loc[i, "time_train_teacher"] = stop_training-start_training
        results.loc[i, "time_test_teacher"] = stop_testing-start_testing

        print(f'Epoch {i:>3}: Training time: {stop_training-start_training:.2f} s, Testing time: {stop_testing-start_testing:.2f} s, Test accuracy: {result:.2f}%')

    end = time()

    print(f'Baseline teacher training time: {end-start:.2f} s')

    """### Train the teacher model and student model on teacher's output"""
    print(
        f"Recreating a teacher with {teacher_num_clauses} clauses and training for {teacher_epochs} epochs on original data")
    teacher_tm = MultiClassTsetlinMachine(
        teacher_num_clauses, T, s, number_of_state_bits=8)

    start = time()

    # train baseline
    for i in range(teacher_epochs):
        start_training = time()
        teacher_tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result = 100*(teacher_tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        results.loc[i, "acc_test_distilled"] = result
        results.loc[i, "time_train_distilled"] = stop_training - \
            start_training
        results.loc[i, "time_test_distilled"] = stop_testing-start_testing

        print(f'Epoch {i:>3}: Training time: {stop_training-start_training:.2f} s, Testing time: {stop_testing-start_testing:.2f} s, Test accuracy: {result:.2f}%')

    print(
        f"Creating a student with {student_num_clauses} clauses and training for {student_epochs} epochs on teacher's output")
    distilled_tm = MultiClassTsetlinMachine(
        student_num_clauses, T, s, number_of_state_bits=8)

    # train student on teacher's output
    for i in range(teacher_epochs, teacher_epochs+student_epochs):
        start_training = time()
        distilled_tm.fit(teacher_tm.transform(X_train),
                       Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result = 100 * \
            (distilled_tm.predict(teacher_tm.transform(X_test)) == Y_test).mean()
        stop_testing = time()

        results.loc[i, "acc_test_distilled"] = result
        results.loc[i, "time_train_distilled"] = stop_training - \
            start_training
        results.loc[i, "time_test_distilled"] = stop_testing-start_testing

        print(f'Epoch {i:>3}: Training time: {stop_training-start_training:.2f} s, Testing time: {stop_testing-start_testing:.2f} s, Test accuracy: {result:.2f}%')

    end = time()

    print(f'Teacher-student training time: {end-start:.2f} s')

    return results


if __name__ == "__main__":
    """### Load CIFAR-10 data"""
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Preprocess data
    X_train = np.copy(X_train)
    X_test = np.copy(X_test)

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[3]):
                X_train[i,:,:,j] = cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 

    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[3]):
            X_test[i,:,:,j] = cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_test = Y_test.reshape(Y_test.shape[0])

    # Input data flattening    
    X_train = X_train.reshape(X_train.shape[0], 32*32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32*32, 3)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    cifar10_params = {
        "teacher_num_clauses": 40000,
        "student_num_clauses": 2000,
        "T": 48000,
        "s": 10,
        "teacher_epochs": 45,
        "student_epochs": 45
    }

    cifar10_results = run_detailed_experiment(
        X_train, Y_train, X_test, Y_test, "CIFAR-10", cifar10_params)
    cifar10_results.to_csv(os.path.join(
        "experiments", "cifar10_results.csv"))
    print(cifar10_results)
    
    """### Load Fashion MNIST data"""
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    # Data booleanization
    X_train = np.where(X_train > 75, 1, 0)
    X_test = np.where(X_test > 75, 1, 0)

    # Input data flattening
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    fashion_mnist_results = run_detailed_experiment(
        X_train, Y_train, X_test, Y_test, "MNIST")
    fashion_mnist_results.to_csv(os.path.join(
        "experiments", "fashion_mnist_results.csv"))
    print(fashion_mnist_results)
    
    """### Load MNIST data"""

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # Data booleanization
    X_train = np.where(X_train > 75, 1, 0)
    X_test = np.where(X_test > 75, 1, 0)

    # Input data flattening
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    mnist_results = run_detailed_experiment(
        X_train, Y_train, X_test, Y_test, "MNIST")
    mnist_results.to_csv(os.path.join(
        "experiments", "mnist_results.csv"))
    print(mnist_results)