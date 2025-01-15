
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from torchvision.datasets import KMNIST
from torchvision import transforms
from stats import entropy, normalize, softmax, joint_probs, mutual_information, kl_divergence
import numpy as np
from time import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from util import save_json
from imbd_ex import prepare_imdb_data
from datetime import datetime

# set seeds
np.random.seed(0)

DEFAULTS = {
    "teacher_num_clauses": 400,
    "student_num_clauses": 200,
    "T": 10,
    "s": 5,
    "teacher_epochs": 60,
    "student_epochs": 60,
    "weighted_clauses": True,
    "number_of_state_bits": 8
}

def distilled_experiment(
    X_train, Y_train, X_test, Y_test,
    experiment_name,
    params=DEFAULTS,
) -> dict:
    """
    Train a baseline student model with teacher_epochs + student_epochs epochs
    Train a baseline teacher model with teacher_epochs + student_epochs epochs
    Rebuild the teacher model and train it with teacher_epochs epochs
    Train the student model with student_epochs epochs on the teacher's output

    """
    exp_start = time()
    # fill in missing parameters with defaults
    for key, value in DEFAULTS.items():
        if key not in params:
            print(f"Parameter {key} not specified, using default value {value}")
            params[key] = value

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
        student_num_clauses, T, s, number_of_state_bits=params["number_of_state_bits"], weighted_clauses=params["weighted_clauses"])

    start = time()

    # train baseline
    for i in range(combined_epochs):
        start_training = time()
        baseline_distilled_tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        prediction, class_sums = baseline_distilled_tm.predict_class_sums_2d(X_test)
        result = 100*(prediction == Y_test).mean()
        stop_testing = time()

        results.loc[i, "acc_test_student"] = result
        results.loc[i, "time_train_student"] = stop_training-start_training
        results.loc[i, "time_test_student"] = stop_testing-start_testing

        print(f'Epoch {i:>3}: Training time: {stop_training-start_training:.2f} s, Testing time: {stop_testing-start_testing:.2f} s, Test accuracy: {result:.2f}%')

    end = time()

    print(f'Baseline student training time: {end-start:.2f} s')

    """### Train the baseline teacher model"""
    print(
        f"Creating a baseline teacher with {teacher_num_clauses} clauses and training on original data")
    baseline_teacher_tm = MultiClassTsetlinMachine(
        teacher_num_clauses, T, s, number_of_state_bits=params["number_of_state_bits"], weighted_clauses=params["weighted_clauses"])

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
        teacher_num_clauses, T, s, number_of_state_bits=params["number_of_state_bits"], weighted_clauses=params["weighted_clauses"])

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
        student_num_clauses, T, s, number_of_state_bits=params["number_of_state_bits"], weighted_clauses=params["weighted_clauses"])

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

    # calculate information transfer

    # now get final prediction probabilities for teacher and distilled
    # class sums are arranged by [sums] x samples, so shape is (num_samples, num_classes)
    student_prediction, student_class_sums = baseline_distilled_tm.predict_class_sums_2d(X_test)
    teacher_prediction, teacher_class_sums = baseline_teacher_tm.predict_class_sums_2d(X_test)
    distilled_prediction, distilled_class_sums = baseline_distilled_tm.predict_class_sums_2d(X_test)

    # class sums are arranged by [sums] x samples
    # output an array here by sample and then avg
    #print(student_class_sums)
    student_probs = np.apply_along_axis(softmax, 1, student_class_sums)
    teacher_probs = np.apply_along_axis(softmax, 1, teacher_class_sums)
    distilled_probs = np.apply_along_axis(softmax, 1, distilled_class_sums)
    #print(student_probs)

    # calculate entropy of each sample, into a 1D array
    # entropy is calculated by -sum(p(x)log(p(x))) for each class for each sample
    student_entropy = np.apply_along_axis(entropy, 1, student_probs)
    teacher_entropy = np.apply_along_axis(entropy, 1, teacher_probs)
    distilled_entropy = np.apply_along_axis(entropy, 1, distilled_probs)

    # calculate kl divergence between teacher and distilled
    #kl_sd = np.apply_along_axis(kl_divergence, 1, student_probs, distilled_probs)
    #kl_st = np.apply_along_axis(kl_divergence, 1, student_probs, teacher_probs)

    # assuming independence, calculate joint probabilities
    joint_probabilities = joint_probs(teacher_probs, distilled_probs)

    # calculate mutual information
    #mi = mutual_information(teacher_probs, distilled_probs, joint_probabilities)
    #print(f"Mutual information: {mi:.4f}")

    # calculate mutual information using sklearn
    mi_sklearn_dt = mutual_info_score(teacher_prediction, distilled_prediction)
    mi_sklearn_ds = mutual_info_score(student_prediction, distilled_prediction)

    print(f"Mutual information (distilled <-> teacher) (sklearn): {mi_sklearn_dt:.4f}")
    print(f"Mutual information (distilled <-> student) (sklearn): {mi_sklearn_ds:.4f}")

    # compute averages for accuracy
    avg_acc_test_teacher = results["acc_test_teacher"].mean()
    std_acc_test_teacher = results["acc_test_teacher"].std()
    avg_acc_test_student = results["acc_test_student"].mean()
    std_acc_test_student = results["acc_test_student"].std()
    avg_acc_test_distilled = results["acc_test_distilled"].mean()
    std_acc_test_distilled = results["acc_test_distilled"].std()

    # compute sum of training times
    sum_time_train_teacher = results["time_train_teacher"].sum()
    sum_time_train_student = results["time_train_student"].sum()
    sum_time_train_distilled = results["time_train_distilled"].sum()
    total_time = time() - exp_start

    output = {
        "results": results.to_dict(),
        "analysis": {
            "avg_acc_test_teacher": avg_acc_test_teacher,
            "std_acc_test_teacher": std_acc_test_teacher,
            "avg_acc_test_student": avg_acc_test_student,
            "std_acc_test_student": std_acc_test_student,
            "avg_acc_test_distilled": avg_acc_test_distilled,
            "std_acc_test_distilled": std_acc_test_distilled,
            "final_acc_test_distilled": results["acc_test_distilled"].iloc[-1],
            "sum_time_train_teacher": sum_time_train_teacher,
            "sum_time_train_student": sum_time_train_student,
            "sum_time_train_distilled": sum_time_train_distilled,
            "total_time": total_time
        },
        "mutual_information": {
            "sklearn_teacher": mi_sklearn_dt,
            "sklearn_student": mi_sklearn_ds
        },
        "params": params,
        "experiment_name": experiment_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": experiment_name + f"_tnc{teacher_num_clauses}_snc{student_num_clauses}_T{T}_s{s}_te{teacher_epochs}_se{student_epochs}"
    }
    
    return output, results


if __name__ == "__main__":
    """### Load KMNIST data"""
    train = KMNIST(root="data", download=True, train=True, transform=transforms.ToTensor())
    test = KMNIST(root="data", download=True, train=False, transform=transforms.ToTensor())

    X_train, Y_train = train.data.numpy(), train.targets.numpy()
    X_test, Y_test = test.data.numpy(), test.targets.numpy()

    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)

    X_train = np.where(X_train > 75, 1, 0)
    X_test = np.where(X_test > 75, 1, 0)

    kmnist_experiments = [
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 600, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 600, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 1600, "student_num_clauses": 400, "T": 600, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 2400, "student_num_clauses": 400, "T": 600, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
    ]
    
    for i, params in enumerate(kmnist_experiments):
        kmnist_results, df = distilled_experiment(
            X_train, Y_train, X_test, Y_test, f"KMNIST_{i}", params)
        save_json(kmnist_results, os.path.join(
            "experiments", f"kmnist_results_{i}.json"))
        df.to_csv(os.path.join(
            "experiments", f"kmnist_results_{i}.csv"))
        print(kmnist_results)

    """### Load CIFAR-10 data"""
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        
    # Input data flattening
    X_train = X_train.reshape(X_train.shape[0], 32*32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32*32, 3)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    cifar10_experiments = [
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 500, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 500, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 1600, "student_num_clauses": 400, "T": 500, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 2400, "student_num_clauses": 400, "T": 500, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
    ]

    for i, params in enumerate(cifar10_experiments):
        cifar10_results, df = distilled_experiment(
            X_train, Y_train, X_test, Y_test, f"CIFAR-10_{i}", params)
        save_json(cifar10_results, os.path.join(
            "experiments", f"cifar10_results_{i}.json"))
        df.to_csv(os.path.join(
            "experiments", f"cifar10_results_{i}.csv"))
        print(cifar10_results)

    """### Load IMDB data"""
    (X_train, Y_train), (X_test, Y_test) = prepare_imdb_data()
    imdb_experiments = [
        {"teacher_num_clauses": 500, "student_num_clauses": 100, "T": 80*100, "s": 10.0, "teacher_epochs": 15, "student_epochs": 15},
        {"teacher_num_clauses": 1000, "student_num_clauses": 250, "T": 80*100, "s": 10.0, "teacher_epochs": 15, "student_epochs": 15},
        {"teacher_num_clauses": 2000, "student_num_clauses": 250, "T": 80*100, "s": 10.0, "teacher_epochs": 15, "student_epochs": 15},
        {"teacher_num_clauses": 3000, "student_num_clauses": 500, "T": 80*100, "s": 10.0, "teacher_epochs": 15, "student_epochs": 15},
        {"teacher_num_clauses": 2000, "student_num_clauses": 1000, "T": 80*100, "s": 10.0, "teacher_epochs": 15, "student_epochs": 15},
    ]
    
    for i, params in enumerate(imdb_experiments):
        imdb_results, df = distilled_experiment(
            X_train, Y_train, X_test, Y_test, f"IMDB_{i}", params)
        save_json(imdb_results, os.path.join(
            "experiments", f"imdb_results_{i}.json"))
        df.to_csv(os.path.join(
            "experiments", f"imdb_results_{i}.csv"))
        print(imdb_results)

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

    mnist_experiments = [
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 1600, "student_num_clauses": 400, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 2400, "student_num_clauses": 400, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
    ]

    for i, params in enumerate(mnist_experiments):
        mnist_results, df = distilled_experiment(
            X_train, Y_train, X_test, Y_test, f"MNIST_{i}", params)
        save_json(mnist_results, os.path.join(
            "experiments", f"mnist_results_{i}.json"))
        df.to_csv(os.path.join(
            "experiments", f"mnist_results_{i}.csv"))
        print(mnist_results)

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
    
    fashion_mnist_experiments = [
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 1600, "student_num_clauses": 400, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 2400, "student_num_clauses": 400, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
    ]
    
    for i, params in enumerate(mnist_experiments):
        fmnist_results, df = distilled_experiment(
            X_train, Y_train, X_test, Y_test, f"Fashion_MNIST_{i}", params)
        save_json(fmnist_results, os.path.join(
            "experiments", f"fashion_mnist_results_{i}.json"))
        df.to_csv(os.path.join(
            "experiments", f"fashion_mnist_results_{i}.csv"))
        print(fmnist_results)


