from stats import entropy, normalize, softmax, joint_probs, mutual_information, kl_divergence
import numpy as np
from time import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from util import save_json, load_pkl, save_pkl, make_dir, rm_file, load_json
from datetime import datetime
from tqdm import tqdm, trange
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from datasets import Dataset
from scipy.interpolate import interp1d

TEACHER_BASELINE_MODEL_PATH = "teacher_baseline.pkl"
STUDENT_BASELINE_MODEL_PATH = "student_baseline.pkl"
TEACHER_CHECKPOINT_PATH = "teacher_checkpoint.pkl"
DISTILLED_MODEL_PATH = "distilled.pkl"

ACC_TEST_TEACHER = "acc_test_teacher"
ACC_TEST_STUDENT = "acc_test_student"
ACC_TEST_DISTILLED = "acc_test_distilled"
TIME_TRAIN_TEACHER = "time_train_teacher"
TIME_TRAIN_STUDENT = "time_train_student"
TIME_TRAIN_DISTILLED = "time_train_distilled"
TIME_TEST_TEACHER = "time_test_teacher"
TIME_TEST_STUDENT = "time_test_student"
TIME_TEST_DISTILLED = "time_test_distilled"

RESULTS_COLUMNS = [ACC_TEST_TEACHER, ACC_TEST_STUDENT, ACC_TEST_DISTILLED, TIME_TRAIN_TEACHER, TIME_TRAIN_STUDENT,
                           TIME_TRAIN_DISTILLED, TIME_TEST_TEACHER, TIME_TEST_STUDENT, TIME_TEST_DISTILLED]

DISTILLED_DEFAULTS = {
    "teacher_num_clauses": 400,
    "student_num_clauses": 200,
    "T": 10,
    "s": 5,
    "teacher_epochs": 60,
    "student_epochs": 60,
    "weighted_clauses": True,
    "number_of_state_bits": 8,
    "downsample": 0
}

DOWNSAMPLE_DEFAULTS = [0.05, 0.10, 0.15, 0.20, 0.25]
def downsample_clauses(X_train_transformed:np.ndarray, X_test_transformed:np.ndarray, downsample: float, symmetric: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Downsample clauses by removing those that are too active or too inactive.

    This function performs clause pruning by removing clauses that activate too frequently or too rarely.
    Clauses that are too active (above over threshold) are considered too specific and not generalizable.
    Clauses that are too inactive (below under threshold) are considered too general and not specific enough.

    Args:
        X_train_transformed (np.ndarray): Training data transformed by teacher TM's clauses
        X_test_transformed (np.ndarray): Test data transformed by teacher TM's clauses  
        downsample (float): Drop clauses that are activated in (1 - downsample)*100% of the time. 
            If downsample is 0.05, then any clause that is activated in 95% of the time is dropped.
        symmetric (bool): If True, drop clauses that are inactive in (1 - downsample)*100% of the time.

    Returns:
        tuple: Contains:
            - X_train_reduced (np.ndarray): Training data with pruned clauses removed
            - X_test_reduced (np.ndarray): Test data with pruned clauses removed
            - num_clauses_dropped (int): Number of clauses that were pruned
    """
    # drop clauses that are too active or too inactive
    # this is a form of data mining where we seek out the clauses that are most informative
    # we drop the clauses that are too active because they are too specific and not generalizable
    # this should reduce the number of clauses and make the student learn faster
    # this works pretty well with downsample = 0.05
    assert downsample >= 0 and downsample < 1, "Downsample should be a float between 0 and 1"
    sums = np.sum(X_train_transformed, axis=0) # shape is (num_classes*num_clauses)
    normalized_sums = sums / X_train_transformed.shape[0] # get the sum of each clause over all samples divided by the number of samples

    # find where to drop
    over_clauses = np.where(normalized_sums > (1 - downsample))[0]
    under_clauses = np.where(normalized_sums < downsample)[0]
    if symmetric:
        clauses_to_drop = np.concatenate([over_clauses, under_clauses])
    else:
        clauses_to_drop = over_clauses

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

def validate_params(params: dict, experiment_name: str) -> str:
    """
    Validate the parameters for the experiment.

    Args:
        params (dict): Parameters for the experiment
        experiment_name (str): Name of the experiment

    Returns:
        str: id of the experiment
    """
    # check that the parameters are valid
    assert "teacher_num_clauses" in params, "Teacher number of clauses not specified"
    assert "T" in params, "T not specified"
    assert "s" in params, "s not specified"
    assert "teacher_epochs" in params, "Teacher epochs not specified"
    assert "student_num_clauses" in params, "Student number of clauses not specified"
    assert "student_epochs" in params, "Student epochs not specified"
    assert params["teacher_num_clauses"] > params["student_num_clauses"], "Student clauses should be less than teacher clauses"
    assert params["downsample"] >= 0 and params["downsample"] <= 1, "Downsample should be a float between 0 and 1"

    params["combined_epochs"] = params["teacher_epochs"] + params["student_epochs"]
    
    # generate experiment id
    exid = f"{experiment_name.replace(' ', '-')}_tnc{params['teacher_num_clauses']}_snc{params['student_num_clauses']}_" \
           f"T{params['T']}_s{params['s']}_te{params['teacher_epochs']}_se{params['student_epochs']}_" \
           f"downsample{params['downsample']}"
    
    return exid


def distilled_experiment(
    dataset: Dataset,
    experiment_name: str,
    params: dict = DISTILLED_DEFAULTS,
    folderpath: str = "experiments",
    save_all: bool = False,
    overwrite: bool = True,
    baseline_teacher_model: MultiClassTsetlinMachine = None,
    baseline_student_model: MultiClassTsetlinMachine = None,
    pretrained_teacher_model: MultiClassTsetlinMachine = None,
    prefilled_results: pd.DataFrame = None,
) -> dict:
    """
    Run a distillation experiment comparing teacher, student, and distilled models.

    Note on baseline_teacher_model and baseline_student_model:
    This is really only for the downsample experiment where we only want to change the downsampling parameters.
    This lets us use the same teacher and student models for all downsampling experiments.
    Remember, the training looks like this:
        student_model trained on original data for combined_epochs
        teacher_model trained on original data for combined_epochs, but a checkpoint is saved after teacher_epochs
        distilled_model trained on output of teacher_model (transformed and downsampled) for student_epochs

    Args:
        dataset (Dataset): The dataset to use for the experiment
        experiment_name (str): Name of the experiment
        params (dict, optional): Parameters for the experiment. Defaults to DISTILLED_DEFAULTS.
        folderpath (str, optional): Path to save experiment results. Defaults to "experiments".
        save_all (bool, optional): Whether to save all models. Defaults to False. If True, saves 
            all models to the experiment directory with paths teacher_baseline.pkl, student_baseline.pkl, distilled.pkl
        baseline_teacher_model (MultiClassTsetlinMachine, optional): The baseline teacher model. Defaults to None.
            If None, the teacher model is trained from scratch. Else, the teacher model is loaded from the given path.
        baseline_student_model (MultiClassTsetlinMachine, optional): The baseline student model. Defaults to None.
            If None, the student model is trained from scratch. Else, the student model is loaded from the given path.
        pretrained_teacher_model (MultiClassTsetlinMachine, optional): The pretrained teacher model. Defaults to None.
            If None, the teacher model is loaded from the given path. Else, the teacher model is trained from scratch.

    Returns:
        dict: Dictionary containing experiment results including:
            - Teacher, student and distilled model accuracies
            - Training and testing times
            - Number of clauses dropped during distillation
            - Total experiment time
    """
    exp_start = time()
    # check that the data is valid
    dataset.validate_lengths()
    X_train, Y_train, X_test, Y_test = dataset.get_data()

    # fill in missing parameters with defaults
    for key, value in DISTILLED_DEFAULTS.items():
        if key not in params:
            print(f"Parameter {key} not specified, using default value {value}")
            params[key] = value

    print(f"Running {experiment_name} with params: {params}")

    # get experiment id
    experiment_id = validate_params(params, experiment_name)
    print(f"Experiment ID: {experiment_id}")

    # create an experiment directory
    if not overwrite and os.path.exists(os.path.join(folderpath, experiment_id)):
        # check if output.json, results.csv, and accuracy.png exist
        if os.path.exists(os.path.join(folderpath, experiment_id, "output.json")) and \
           os.path.exists(os.path.join(folderpath, experiment_id, "results.csv")) and \
           os.path.exists(os.path.join(folderpath, experiment_id, "accuracy.png")):
            print(f"Experiment {experiment_id} already exists, skipping")
            # load the results
            results = pd.read_csv(os.path.join(folderpath, experiment_id, "results.csv"))
            output = load_json(os.path.join(folderpath, experiment_id, "output.json"))
            return output, results

    make_dir(os.path.join(folderpath, experiment_id), overwrite=True)
    teacher_model_path = os.path.join(folderpath, experiment_id, TEACHER_CHECKPOINT_PATH)

    # create models
    baseline_student_tm = MultiClassTsetlinMachine(
        params['student_num_clauses'], params['T'], params['s'], number_of_state_bits=params["number_of_state_bits"], weighted_clauses=params["weighted_clauses"])
    baseline_teacher_tm = MultiClassTsetlinMachine(
        params['teacher_num_clauses'], params['T'], params['s'], number_of_state_bits=params["number_of_state_bits"], weighted_clauses=params["weighted_clauses"])
    distilled_tm = MultiClassTsetlinMachine(
        params['student_num_clauses'], params['T'], params['s'], number_of_state_bits=params["number_of_state_bits"], weighted_clauses=params["weighted_clauses"])

    # create a results dataframe
    if prefilled_results is None: 
        results = pd.DataFrame(columns=RESULTS_COLUMNS, index=range(params['combined_epochs']))
    else:
        results = prefilled_results
        assert results.columns.tolist() == RESULTS_COLUMNS, "Prefilled results columns do not match expected columns"
        assert results.index._range == range(params['combined_epochs']), "Prefilled results index does not match expected index"

        # delete entries after params['teacher_epochs'] for 'acc_test_distilled', 'time_train_distilled', 'time_test_distilled' columns
        results.loc[params['teacher_epochs']:, ACC_TEST_DISTILLED] = np.nan
        results.loc[params['teacher_epochs']:, TIME_TRAIN_DISTILLED] = np.nan 
        results.loc[params['teacher_epochs']:, TIME_TEST_DISTILLED] = np.nan
        print(f"Prefilled results loaded")

    # train baselines
    # train baseline student
    if isinstance(baseline_student_model, MultiClassTsetlinMachine):
        print(f"Loading pretrained baseline student model")
        baseline_student_tm = baseline_student_model
    else:
        print(f"Creating a baseline student with {params['student_num_clauses']} clauses and training on original data")
        start = time()
        bs_pbar = tqdm(range(params['combined_epochs']), desc="Student", leave=False, dynamic_ncols=True)
        for i in bs_pbar:
            result, train_time, test_time = train_step(baseline_student_tm, X_train, Y_train, X_test, Y_test)
            results.loc[i, ACC_TEST_STUDENT], results.loc[i, TIME_TRAIN_STUDENT], results.loc[i, TIME_TEST_STUDENT] = result, train_time, test_time
            tqdm.write(f'Epoch {i:>3}: Training time: {train_time:.2f} s, Testing time: {test_time:.2f} s, Test accuracy: {result:.2f}%')
            bs_pbar.set_description(f"Student: {results[ACC_TEST_STUDENT].mean():.2f}%")

        bs_pbar.close()
        end = time()
        print(f'Baseline student training time: {end-start:.2f} s')

    # train baseline teacher
    if isinstance(baseline_teacher_model, MultiClassTsetlinMachine):
        print(f"Loading pretrained baseline teacher model")
        baseline_teacher_tm = baseline_teacher_model
    else:
        print(f"Creating a baseline teacher with {params['teacher_num_clauses']} clauses and training on original data")
        start = time()
        bt_pbar = tqdm(range(params['combined_epochs']), desc="Teacher", leave=False, dynamic_ncols=True)
        for i in bt_pbar:
            result, train_time, test_time = train_step(baseline_teacher_tm, X_train, Y_train, X_test, Y_test)
            results.loc[i, ACC_TEST_TEACHER], results.loc[i, TIME_TRAIN_TEACHER], results.loc[i, TIME_TEST_TEACHER] = result, train_time, test_time
            tqdm.write(f'Epoch {i:>3}: Training time: {train_time:.2f} s, Testing time: {test_time:.2f} s, Test accuracy: {result:.2f}%')
            bt_pbar.set_description(f"Teacher: {results[ACC_TEST_TEACHER].mean():.2f}%")

            if i == params['teacher_epochs'] - 1:
                save_pkl(baseline_teacher_tm, teacher_model_path)
                tqdm.write(f"Saved teacher model to {teacher_model_path}")

        bt_pbar.close()
        end = time()
        print(f'Baseline teacher training time: {end-start:.2f} s')

    # copy first teacher_epochs results to distilled results
    results.loc[:params['teacher_epochs'], ACC_TEST_DISTILLED] = results.loc[:params['teacher_epochs'], ACC_TEST_TEACHER]
    results.loc[:params['teacher_epochs'], TIME_TRAIN_DISTILLED] = results.loc[:params['teacher_epochs'], TIME_TRAIN_TEACHER]
    results.loc[:params['teacher_epochs'], TIME_TEST_DISTILLED] = results.loc[:params['teacher_epochs'], TIME_TEST_TEACHER]

    # train distilled model
    if isinstance(pretrained_teacher_model, MultiClassTsetlinMachine):
        print(f"Loading pretrained teacher model")
        teacher_tm = pretrained_teacher_model
    else:
        print(f"Loading teacher model from {teacher_model_path}, trained for {params['teacher_epochs']} epochs")
        teacher_tm = load_pkl(teacher_model_path)
        if not save_all:
            rm_file(teacher_model_path) # remove the teacher model file. we don't need it anymore

    # downsample clauses
    X_train_transformed = teacher_tm.transform(X_train)
    X_test_transformed = teacher_tm.transform(X_test)
    X_train_downsampled, X_test_downsampled, num_clauses_dropped = downsample_clauses(X_train_transformed, X_test_transformed, params['downsample'], symmetric=True)
    reduction_percentage = 100*(num_clauses_dropped/X_train_transformed.shape[1]) # calculate the percentage of clauses dropped
    if num_clauses_dropped == X_train_transformed.shape[1]:
        print(f"Every clause was dropped, skipping distillation")
        return None, results

    start = time()
    print(f"Training distilled model for {params['student_epochs']} epochs")
    dt_pbar = tqdm(range(params['teacher_epochs'], params['combined_epochs']), desc="Distilled", leave=False, dynamic_ncols=True)
    for i in dt_pbar:
        result, train_time, test_time = train_step(distilled_tm, X_train_downsampled, Y_train, X_test_downsampled, Y_test)
        results.loc[i, ACC_TEST_DISTILLED], results.loc[i, TIME_TRAIN_DISTILLED], results.loc[i, TIME_TEST_DISTILLED] = result, train_time, test_time

        tqdm.write(f'Epoch {i:>3}: Training time: {train_time:.2f} s, Testing time: {test_time:.2f} s, Test accuracy: {result:.2f}%')
        dt_pbar.set_description(f"Distilled: {results[ACC_TEST_DISTILLED].mean():.2f}%")

    dt_pbar.close()
    end = time()

    print(f'Teacher-student training time: {end-start:.2f} s')
    # calculate information transfer
    # now get final prediction probabilities for teacher and distilled
    # class sums are arranged by [sums] x samples, so shape is (num_samples, num_classes)
    student_prediction, student_class_sums = baseline_student_tm.predict_class_sums_2d(X_test)
    teacher_prediction, teacher_class_sums = baseline_teacher_tm.predict_class_sums_2d(X_test)
    distilled_prediction, distilled_class_sums = baseline_student_tm.predict_class_sums_2d(X_test)

    # calculate mutual information using sklearn
    mi_sklearn_dt = mutual_info_score(teacher_prediction, distilled_prediction)
    mi_sklearn_ds = mutual_info_score(student_prediction, distilled_prediction)

    print(f"Mutual information (distilled <-> teacher) (sklearn): {mi_sklearn_dt:.4f}")
    print(f"Mutual information (distilled <-> student) (sklearn): {mi_sklearn_ds:.4f}")

    # compute our mutual information with L/C*log(L/C)
    L_student = X_train.shape[1] # number of literals for the student
    L_teacher = X_train.shape[1] # number of literals for the teacher
    L_distilled = X_train_transformed.shape[1] # number of literals for the distilled
    C_student = params['student_num_clauses'] # number of clauses for the student
    C_teacher = params['teacher_num_clauses'] # number of clauses for the teacher
    C_distilled = params['student_num_clauses'] # number of clauses for the distilled
    
    info_teacher = L_teacher/C_teacher*np.log(L_teacher/C_teacher)
    info_student = L_student/C_student*np.log(L_student/C_student)
    info_distilled = L_distilled/C_distilled*np.log(L_distilled/C_distilled)

    print(f"Information (distilled <-> teacher) (L/C*log(L/C)): {info_teacher:.4f}")
    print(f"Information (distilled <-> student) (L/C*log(L/C)): {info_student:.4f}")
    print(f"Information (distilled <-> distilled) (L/C*log(L/C)): {info_distilled:.4f}")

    total_time = time() - exp_start

    output = {
        "analysis": {
            "avg_acc_test_teacher": results[ACC_TEST_TEACHER].mean(),
            "std_acc_test_teacher": results[ACC_TEST_TEACHER].std(),
            "avg_acc_test_student": results[ACC_TEST_STUDENT].mean(),
            "std_acc_test_student": results[ACC_TEST_STUDENT].std(),
            "avg_acc_test_distilled": results[ACC_TEST_DISTILLED].mean(),
            "std_acc_test_distilled": results[ACC_TEST_DISTILLED].std(),
            "final_acc_test_distilled": results[ACC_TEST_DISTILLED].iloc[-1],
            "final_acc_test_teacher": results[ACC_TEST_TEACHER].iloc[-1],
            "final_acc_test_student": results[ACC_TEST_STUDENT].iloc[-1],
            "sum_time_train_teacher": results[TIME_TRAIN_TEACHER].sum(),
            "sum_time_train_student": results[TIME_TRAIN_STUDENT].sum(),
            "sum_time_train_distilled": results[TIME_TRAIN_DISTILLED].sum(),
            "sum_time_test_teacher": results[TIME_TEST_TEACHER].sum(),
            "sum_time_test_student": results[TIME_TEST_STUDENT].sum(),
            "sum_time_test_distilled": results[TIME_TEST_DISTILLED].sum(),
            "total_time": total_time,
            "num_clauses_dropped": num_clauses_dropped,
            "num_clauses_dropped_percentage": reduction_percentage
        },
        "mutual_information": {
            "sklearn_teacher": mi_sklearn_dt,
            "sklearn_student": mi_sklearn_ds,
            "info_teacher": info_teacher,
            "info_student": info_student,
            "info_distilled": info_distilled
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
    if save_all:
        save_pkl(baseline_teacher_tm, os.path.join(folderpath, experiment_id, TEACHER_BASELINE_MODEL_PATH))
        save_pkl(baseline_student_tm, os.path.join(folderpath, experiment_id, STUDENT_BASELINE_MODEL_PATH))
        save_pkl(distilled_tm, os.path.join(folderpath, experiment_id, DISTILLED_MODEL_PATH))

    # plot results and save
    plt.figure(figsize=(8,6), dpi=300)
    plt.plot(results[ACC_TEST_DISTILLED], label="Distilled")
    plt.plot(results[ACC_TEST_TEACHER], label="Teacher", alpha=0.5)
    plt.plot(results[ACC_TEST_STUDENT], label="Student", alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    #plt.title(f"Testing Accuracy of {experiment_name} Over {combined_epochs} Epochs")
    plt.xticks(range(0, len(results), 5))
    plt.legend(loc="upper left")
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(fpath, "accuracy.png"))
    plt.close()

    return output, results


def downsample_experiment(
    dataset: Dataset,
    experiment_name,
    params=DISTILLED_DEFAULTS,
    downsamples=DOWNSAMPLE_DEFAULTS,
    folderpath="experiments",
    overwrite: bool = True,
) -> dict:
    """
    Run a downsample experiment comparing teacher, student, and distilled models.


    Args:
        X_train (np.ndarray): Training data features
        Y_train (np.ndarray): Training data labels
        X_test (np.ndarray): Test data features 
        Y_test (np.ndarray): Test data labels
        experiment_name (str): Name of the experiment
        params (dict, optional): Parameters for the experiment. Defaults to DOWNSAMPLE_DEFAULTS.
        folderpath (str, optional): Path to save experiment results. Defaults to "experiments".
        save_all (bool, optional): Whether to save all models. Defaults to False.

    Returns:
        dict: Dictionary containing experiment results including:
            - Teacher, student and distilled model accuracies
            - Training and testing times
            - Number of clauses dropped during distillation
            - Total experiment time
    """
    downsamples = sorted(list(set(downsamples))) # remove duplicates and sort
    print(f"Running Downsample Experiment {experiment_name} with params: {params} and downsamples: {downsamples}")
    
    # train distilled model first, fully raw, NO baseline models and NO downsampling
    params["downsample"] = 0
    print("Training distilled model first, fully raw, NO baseline models and NO downsampling")
    subfolderpath = os.path.join(folderpath, experiment_name)
    make_dir(subfolderpath, overwrite=overwrite)
    original_output, original_results_pd = distilled_experiment(dataset, "ds", params, subfolderpath, save_all=True)
    original_id = original_output["id"]

    # now go load each model
    print("Loading baseline models and pretrained teacher model...")
    teacher_model = load_pkl(os.path.join(subfolderpath, original_id, TEACHER_BASELINE_MODEL_PATH))
    student_model = load_pkl(os.path.join(subfolderpath, original_id, STUDENT_BASELINE_MODEL_PATH))
    pretrained_teacher_model = load_pkl(os.path.join(subfolderpath, original_id, TEACHER_CHECKPOINT_PATH))
    print("Done loading baseline models and pretrained teacher model")

    # then train distilled models with downsampling
    all_outputs = []
    print("Training distilled models with downsampling...")
    ended_at = -1
    for i, downsample in enumerate(downsamples):
        print(f"Training distilled model with downsampling {downsample}")
        params["downsample"] = downsample
        output_dict, _ = distilled_experiment(dataset, "ds", params, subfolderpath, 
                                               baseline_teacher_model=teacher_model, 
                                               baseline_student_model=student_model, 
                                               pretrained_teacher_model=pretrained_teacher_model, 
                                               prefilled_results=original_results_pd,
                                               overwrite=overwrite,
                                               save_all=False)
        if output_dict is not None:
            all_outputs.append(output_dict)
        else:
            print(f"Skipping downsample {downsample} because it failed")
            ended_at = i
            break
    
    if ended_at != -1:
        downsamples = downsamples[:ended_at]
        all_outputs = all_outputs[:ended_at]

    print("Done training distilled models with downsampling")

    # now plot the results. Y value is average accuracy of distilled model plotted over different downsamples
    # add a line for baseline teacher and baseline student
    baseline_student_avg_acc = original_output["analysis"]["avg_acc_test_student"]
    baseline_teacher_avg_acc = original_output["analysis"]["avg_acc_test_teacher"]
    baseline_student_final_acc = original_output["analysis"]["final_acc_test_student"]
    baseline_teacher_final_acc = original_output["analysis"]["final_acc_test_teacher"]
    
    all_final_acc = np.array([output["analysis"]["final_acc_test_distilled"] for output in all_outputs])
    all_avg_acc = np.array([output["analysis"]["avg_acc_test_distilled"] for output in all_outputs])
    downsamples = np.array(downsamples)

    interpolator_final = interp1d(downsamples, all_final_acc, kind = "cubic")
    interpolator_avg = interp1d(downsamples, all_avg_acc, kind = "cubic")
    
    # Plotting the Graph
    X_=np.linspace(downsamples.min(), downsamples.max(), 500)
    interp_final_acc = interpolator_final(X_)
    interp_avg_acc = interpolator_avg(X_)
    
    plt.figure(figsize=(8,6), dpi=300)
    plt.axhline(y=baseline_teacher_final_acc, linestyle=':', color="orange", alpha=0.7, label="Final Teacher Accuracy")
    plt.axhline(y=baseline_student_final_acc, linestyle=':', color="green", alpha=0.7, label="Final Student Accuracy")
    #plt.plot(X_, interp_final_acc, label="Final Distilled")
    #plt.plot(X_, interp_avg_acc, label="Avg Distilled")
    plt.plot(downsamples, all_final_acc, marker='o', label="Final Distilled Accuracy")
    plt.xlabel("Downsample Rate")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(subfolderpath, "downsample_results_final_acc.png"))
    plt.close()

    
    plt.figure(figsize=(8,6), dpi=300)
    plt.axhline(y=baseline_teacher_avg_acc, linestyle=':', color="orange", alpha=0.7, label="Avg Teacher Accuracy")
    plt.axhline(y=baseline_student_avg_acc, linestyle=':', color="green", alpha=0.7, label="Avg Student Accuracy")
    #plt.plot(X_, interp_final_acc, label="Final Distilled")
    #plt.plot(X_, interp_avg_acc, label="Avg Distilled")
    plt.plot(downsamples, all_avg_acc, marker='o', label="Avg Distilled Accuracy")
    plt.xlabel("Downsample Rate")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(subfolderpath, "downsample_results_avg_acc.png"))
    plt.close()

    return all_outputs