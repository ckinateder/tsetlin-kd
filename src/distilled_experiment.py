from stats import entropy, normalize, softmax, joint_probs, mutual_information, kl_divergence
import numpy as np
from time import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from util import save_json, load_pkl, save_pkl, make_dir, rm_file
from datetime import datetime
from tqdm import tqdm, trange
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine

DEFAULTS = {
    "teacher_num_clauses": 400,
    "student_num_clauses": 200,
    "T": 10,
    "s": 5,
    "teacher_epochs": 60,
    "student_epochs": 60,
    "weighted_clauses": True,
    "number_of_state_bits": 8,
    "over": 0.95,
    "under": 0.05
}
def downsample_clauses(X_train_transformed:np.ndarray, X_test_transformed:np.ndarray,
                        over: float, under: float):
    """
    Transform the data from the teacher's output into a reduced space
    """
    # drop clauses that are too active or too inactive
    # this is a form of data mining where we seek out the clauses that are most informative
    # we drop the clauses that are too active because they are too specific and not generalizable
    # we drop the clauses that are too inactive because they are too general and not specific enough
    # this should reduce the number of clauses and make the student learn faster
    # this works pretty well with over = 0.95 and under = 0.0

    sums = np.sum(X_train_transformed, axis=0) # shape is (num_classes*num_clauses)
    normalized_sums = sums / X_train_transformed.shape[0] # get the sum of each clause over all samples divided by the number of samples

    # find where to drop
    over_clauses = np.where(normalized_sums > over)[0]
    under_clauses = np.where(normalized_sums < under)[0]
    clauses_to_drop = np.concatenate([over_clauses, under_clauses])

    X_train_reduced = np.delete(X_train_transformed, clauses_to_drop, axis=1) # delete the clauses from the training data
    X_test_reduced = np.delete(X_test_transformed, clauses_to_drop, axis=1) # delete the clauses from the testing data

    num_clauses_dropped = len(clauses_to_drop)
    reduction_percentage = 100*(num_clauses_dropped/X_train_transformed.shape[1]) # calculate the percentage of clauses dropped

    print(f"Dropped {num_clauses_dropped} clauses from {X_train_transformed.shape[1]} clauses, {reduction_percentage:.2f}% reduction")
    
    return X_train_reduced, X_test_reduced, num_clauses_dropped

def train_step(model, X_train, Y_train, X_test, Y_test) -> tuple[float, float, float]:
    """
    Train a model for one epoch and return test accuracy and timing information.

    Args:
        model: The TsetlinMachine model to train
        X_train: Training data features
        Y_train: Training data labels 
        X_test: Test data features
        Y_test: Test data labels

    Returns:
        tuple[float, float, float]: Test accuracy percentage, training time, and testing time
    """
    start_training = time()
    model.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    prediction = model.predict(X_test)
    result = 100*(prediction == Y_test).mean()
    stop_testing = time()

    return result, stop_training-start_training, stop_testing-start_testing

def distilled_experiment(
    X_train, Y_train, X_test, Y_test,
    experiment_name,
    params=DEFAULTS,
    folderpath="experiments",
    save_all=False
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

    # check that the data is valid
    assert len(X_train) == len(Y_train), "Training data length mismatch"
    assert len(X_test) == len(Y_test), "Testing data length mismatch"

    # check that the parameters are valid
    assert "teacher_num_clauses" in params, "Teacher number of clauses not specified"
    assert "T" in params, "T not specified"
    assert "s" in params, "s not specified"
    assert "teacher_epochs" in params, "Teacher epochs not specified"
    assert "student_num_clauses" in params, "Student number of clauses not specified"
    assert "student_epochs" in params, "Student epochs not specified"
    assert params["teacher_num_clauses"] > params["student_num_clauses"], "Student clauses should be less than teacher clauses"
    assert params["over"] >= 0 and params["over"] <= 1, "Over should be a float between 0 and 1"
    assert params["under"] >= 0 and params["under"] <= 1, "Under should be a float between 0 and 1"

    # extract parameters
    teacher_num_clauses = params["teacher_num_clauses"]
    T = params["T"]
    s = params["s"]
    teacher_epochs = params["teacher_epochs"]
    student_num_clauses = params["student_num_clauses"]
    student_epochs = params["student_epochs"]
    combined_epochs = teacher_epochs + student_epochs
    over = params["over"]
    under = params["under"]

    # generate experiment id
    experiment_id = f"{experiment_name}_tnc{teacher_num_clauses}_snc{student_num_clauses}_T{T}_s{s}_te{teacher_epochs}_se{student_epochs}_over{over}_under{under}"
    print(f"Experiment ID: {experiment_id}")

    # create an experiment directory
    make_dir(os.path.join(folderpath, experiment_id), overwrite=True)

    # create a results dataframe
    results = pd.DataFrame(columns=["acc_test_teacher", "acc_test_student", "acc_test_distilled", "time_train_teacher", "time_train_student",
                           "time_train_distilled", "time_test_teacher", "time_test_student", "time_test_distilled"], index=range(combined_epochs))

    """### Train the baseline student model"""
    print(
        f"Creating a baseline student with {student_num_clauses} clauses and training on original data")
    baseline_distilled_tm = MultiClassTsetlinMachine(
        student_num_clauses, T, s, number_of_state_bits=params["number_of_state_bits"], weighted_clauses=params["weighted_clauses"])

    start = time()

    # train baseline student
    bs_pbar = tqdm(range(combined_epochs), desc="Student", leave=False, dynamic_ncols=True)
    for i in bs_pbar:
        result, train_time, test_time = train_step(baseline_distilled_tm, X_train, Y_train, X_test, Y_test)
        results.loc[i, "acc_test_student"], results.loc[i, "time_train_student"], results.loc[i, "time_test_student"] = result, train_time, test_time
        tqdm.write(f'Epoch {i:>3}: Training time: {train_time:.2f} s, Testing time: {test_time:.2f} s, Test accuracy: {result:.2f}%')
        bs_pbar.set_description(f"Student: {results['acc_test_student'].mean():.2f}%")

    bs_pbar.close()
    end = time()

    print(f'Baseline student training time: {end-start:.2f} s')
    if save_all:
        save_pkl(baseline_distilled_tm, os.path.join(folderpath, experiment_id, "student_baseline.pkl"))

    # train baseline teacher
    print(
        f"Creating a baseline teacher with {teacher_num_clauses} clauses and training on original data")
    baseline_teacher_tm = MultiClassTsetlinMachine(
        teacher_num_clauses, T, s, number_of_state_bits=params["number_of_state_bits"], weighted_clauses=params["weighted_clauses"])

    start = time()
    teacher_model_path = os.path.join(folderpath, experiment_id, "teacher_checkpoint.pkl")
    bt_pbar = tqdm(range(combined_epochs), desc="Teacher", leave=False, dynamic_ncols=True)
    for i in bt_pbar:
        result, train_time, test_time = train_step(baseline_teacher_tm, X_train, Y_train, X_test, Y_test)
        results.loc[i, "acc_test_teacher"], results.loc[i, "time_train_teacher"], results.loc[i, "time_test_teacher"] = result, train_time, test_time
        tqdm.write(f'Epoch {i:>3}: Training time: {train_time:.2f} s, Testing time: {test_time:.2f} s, Test accuracy: {result:.2f}%')
        bt_pbar.set_description(f"Teacher: {results['acc_test_teacher'].mean():.2f}%")

        if i == teacher_epochs - 1:
            save_pkl(baseline_teacher_tm, teacher_model_path)
            tqdm.write(f"Saved teacher model to {teacher_model_path}")

    bt_pbar.close()
    end = time()

    # copy first teacher_epochs results to distilled results
    results.loc[:teacher_epochs, "acc_test_distilled"] = results.loc[:teacher_epochs, "acc_test_teacher"]
    results.loc[:teacher_epochs, "time_train_distilled"] = results.loc[:teacher_epochs, "time_train_teacher"]
    results.loc[:teacher_epochs, "time_test_distilled"] = results.loc[:teacher_epochs, "time_test_teacher"]

    print(f'Baseline teacher training time: {end-start:.2f} s')
    if save_all:
        save_pkl(baseline_teacher_tm, os.path.join(folderpath, experiment_id, "teacher_baseline.pkl"))

    # train distilled model
    print(f"Loading teacher model from {teacher_model_path}, trained for {teacher_epochs} epochs")
    teacher_tm = load_pkl(teacher_model_path)
    rm_file(teacher_model_path) # remove the teacher model file. we don't need it anymore

    distilled_tm = MultiClassTsetlinMachine(
        student_num_clauses, T, s, number_of_state_bits=params["number_of_state_bits"], weighted_clauses=params["weighted_clauses"])

    # downsample clauses
    X_train_transformed = teacher_tm.transform(X_train)
    X_test_transformed = teacher_tm.transform(X_test)
    X_train_downsampled, X_test_downsampled, num_clauses_dropped = downsample_clauses(X_train_transformed, X_test_transformed, over, under)
    reduction_percentage = 100*(num_clauses_dropped/X_train_transformed.shape[1]) # calculate the percentage of clauses dropped

    start = time()
    print(f"Training distilled model for {student_epochs} epochs")
    dt_pbar = tqdm(range(teacher_epochs, combined_epochs), desc="Distilled", leave=False, dynamic_ncols=True)
    for i in dt_pbar:
        result, train_time, test_time = train_step(distilled_tm, X_train_downsampled, Y_train, X_test_downsampled, Y_test)
        results.loc[i, "acc_test_distilled"], results.loc[i, "time_train_distilled"], results.loc[i, "time_test_distilled"] = result, train_time, test_time

        tqdm.write(f'Epoch {i:>3}: Training time: {train_time:.2f} s, Testing time: {test_time:.2f} s, Test accuracy: {result:.2f}%')
        dt_pbar.set_description(f"Distilled: {results['acc_test_distilled'].mean():.2f}%")

    dt_pbar.close()
    end = time()

    print(f'Teacher-student training time: {end-start:.2f} s')
    if save_all:
        save_pkl(distilled_tm, os.path.join(folderpath, experiment_id, "distilled.pkl"))

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
            "total_time": total_time,
            "num_clauses_dropped": num_clauses_dropped,
            "num_clauses_dropped_percentage": reduction_percentage
        },
        "mutual_information": {
            "sklearn_teacher": mi_sklearn_dt,
            "sklearn_student": mi_sklearn_ds
        },
        "params": params,
        "experiment_name": experiment_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": experiment_id,
        "results": results.to_dict(),
    }

    # save output
    fpath = os.path.join(folderpath, experiment_id)
    save_json(output, os.path.join(fpath, "output.json"))
    results.to_csv(os.path.join(fpath, "results.csv"))

    # plot results and save
    plt.figure(figsize=(8,6), dpi=300)
    plt.plot(results["acc_test_distilled"], label="Distilled")
    plt.plot(results["acc_test_teacher"], label="Teacher", alpha=0.5)
    plt.plot(results["acc_test_student"], label="Student", alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    #plt.title(f"Testing Accuracy of {experiment_name} Over {combined_epochs} Epochs")
    plt.xticks(range(0, len(results), 5))
    plt.legend(loc="upper left")
    plt.grid(linestyle='dotted')
    # add text of parameters for teacher_num_clauses, student_num_clauses, teacher_epochs, student_epochs, over, under
    params_text = (
        f"teacher_num_clauses: {teacher_num_clauses}\n"
        f"student_num_clauses: {student_num_clauses}\n"
        f"teacher_epochs: {teacher_epochs}\n"
        f"student_epochs: {student_epochs}\n"
        f"over: {over}\n"
        f"under: {under}\n"
    )
    #plt.gcf().text(0.68, 0.14, params_text, fontsize=8, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=1))
    plt.savefig(os.path.join(fpath, "accuracy.png"))
    plt.close()

    return output, results