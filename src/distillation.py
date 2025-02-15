#from stats import entropy, normalize, softmax, joint_probs, mutual_information, kl_divergence
import numpy as np
from time import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from util import save_json, load_pkl, save_pkl, make_dir, rm_file, load_json
from datetime import datetime
from tqdm import tqdm
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from datasets import Dataset
from __init__ import *

def get_downsample_indices(X:np.ndarray, downsample: float, symmetric: bool = True) -> np.ndarray:
    """
    Downsample clauses by removing those that are too active or too inactive.

    This function performs clause pruning by removing clauses that activate too frequently or too rarely.
    Clauses that are too active (above over threshold) are considered too specific and not generalizable.
    Clauses that are too inactive (below under threshold) are considered too general and not specific enough.

    This function returns the *indices* of where to drop - it does not actually drop the clauses

    Args:
        X (np.ndarray): Data transformed by teacher TM's clauses
        downsample (float): Drop clauses that are activated in (1 - downsample)*100% of the time. 
            If downsample is 0.05, then any clause that is activated in 95% of the time is dropped.
        symmetric (bool): If True, ALSO drop clauses that are inactive in (1 - downsample)*100% of the time.
            This doesn't usually make a difference to the distilled model's performance.

    Returns:
        np.ndarray: Indices of clauses to drop
    """
    # drop clauses that are too active or too inactive
    # this is a form of data mining where we seek out the clauses that are most informative
    # we drop the clauses that are too active because they are too specific and not generalizable
    # this should reduce the number of clauses and make the student learn faster
    # this works pretty well with downsample = 0.05
    assert downsample >= 0 and downsample < 1, "Downsample should be a float between 0 and 1"
    sums = np.sum(X, axis=0) # shape is (num_classes*num_clauses)
    normalized_sums = sums / X.shape[0] # get the sum of each clause over all samples divided by the number of samples

    # find where to drop
    over_clauses = np.where(normalized_sums > (1 - downsample))[0] # clauses that are activated in (1 - downsample)*100% of the time
    under_clauses = np.where(normalized_sums < downsample)[0] # clauses that are inactive (1 - downsample)*100% of the time
    if symmetric:
        clauses_to_drop = np.concatenate([over_clauses, under_clauses])
    else:
        clauses_to_drop = over_clauses

    return clauses_to_drop

def downsample_clauses(X_train_transformed:np.ndarray, X_test_transformed:np.ndarray, downsample: float, symmetric: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
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
        symmetric (bool): If True, ALSO drop clauses that are inactive in (1 - downsample)*100% of the time.
            This doesn't usually make a difference to the distilled model's performance.

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
    clauses_to_drop = get_downsample_indices(X_train_transformed, downsample, symmetric=symmetric)

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


def distillation_experiment(
    dataset: Dataset,
    experiment_name: str,
    params: dict = DISTILLED_DEFAULTS,
    folderpath: str = DEFAULT_FOLDERPATH,
    save_all: bool = False,
    overwrite: bool = False,
    baseline_teacher_model: MultiClassTsetlinMachine | None = None,
    baseline_student_model: MultiClassTsetlinMachine | None = None,
    pretrained_teacher_model: MultiClassTsetlinMachine | None = None,
    prefilled_results: pd.DataFrame | None = None,
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
        folderpath (str, optional): Path to save experiment results. Defaults to DEFAULT_FOLDERPATH.
        save_all (bool, optional): Whether to save all models. Defaults to False. If True, saves 
            all models to the experiment directory with paths teacher_baseline.pkl, student_baseline.pkl, distilled.pkl
        baseline_teacher_model (MultiClassTsetlinMachine, optional): The baseline teacher model. Defaults to None.
            If None, the teacher model is trained from scratch. Else, the teacher model is loaded from the given path.
        baseline_student_model (MultiClassTsetlinMachine, optional): The baseline student model. Defaults to None.
            If None, the student model is trained from scratch. Else, the student model is loaded from the given path.
        pretrained_teacher_model (MultiClassTsetlinMachine, optional): The pretrained teacher model. Defaults to None.
            If None, the teacher model is loaded from the given path. Else, the teacher model is trained from scratch.

    Returns:
        tuple: Tuple containing:
            - Dictionary containing experiment results including:
                - Teacher, student and distilled model accuracies
                - Training and testing times
                - Number of clauses dropped during distillation
                - Total experiment time
            - pd.DataFrame: Results dataframe
    """
    exp_start = time()
    print(f"Starting experiment {experiment_name} at {exp_start}")
    # check that the data is valid
    dataset.validate_lengths()
    X_train, Y_train, X_test, Y_test = dataset.get_data()

    # fill in missing parameters with defaults
    for key, value in DISTILLED_DEFAULTS.items():
        if key not in params:
            print(f"Parameter {key} not specified, using default value {value}")
            params[key] = value

    print(f"Using params: {params}")

    # get experiment id
    experiment_id = validate_params(params, experiment_name)
    print(f"Experiment ID: {experiment_id}")

    # create an experiment directory
    if not overwrite and os.path.exists(os.path.join(folderpath, experiment_id)):
        # Check if experiment files exist, and if save_all, check model files exist too
        basic_files_exist = all(os.path.exists(os.path.join(folderpath, experiment_id, f)) 
                              for f in [OUTPUT_JSON_PATH, RESULTS_CSV_PATH, ACCURACY_PNG_PATH])
        
        model_files_exist = not save_all or all(os.path.exists(os.path.join(folderpath, experiment_id, f)) 
                                               for f in [TEACHER_BASELINE_MODEL_PATH, 
                                                       STUDENT_BASELINE_MODEL_PATH,
                                                       DISTILLED_MODEL_PATH])
        
        if basic_files_exist and model_files_exist:
            print(f"Experiment {experiment_id} already exists, skipping")
            # load the results
            results = pd.read_csv(os.path.join(folderpath, experiment_id, RESULTS_CSV_PATH))
            output = load_json(os.path.join(folderpath, experiment_id, OUTPUT_JSON_PATH))
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
        assert results.columns.tolist() == RESULTS_COLUMNS, \
            f"Prefilled results columns do not match expected columns: {results.columns.tolist()} != {RESULTS_COLUMNS}"
        assert results.index.to_list() == list(range(params['combined_epochs'])), \
            f"Prefilled results index does not match expected index: {results.index.to_list()} != {range(params['combined_epochs'])}"

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
    print(f"Getting offline clause outputs from teacher model")
    X_train_transformed = teacher_tm.transform(X_train)
    X_test_transformed = teacher_tm.transform(X_test)

    print(f"Downsampling clauses with downsample rate {params['downsample']}")
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

    print(f"Calculating information transfer")
    # calculate information transfer
    # get outputs
    student_prediction, student_class_sums = baseline_student_tm.predict_class_sums_2d(X_test)
    teacher_prediction, teacher_class_sums = baseline_teacher_tm.predict_class_sums_2d(X_test)
    distilled_prediction, distilled_class_sums = distilled_tm.predict_class_sums_2d(X_test_transformed)

    # sklearn mutual information
    mi_true_student = mutual_info_score(Y_test, student_prediction)
    mi_true_teacher = mutual_info_score(Y_test, teacher_prediction)
    mi_true_distilled = mutual_info_score(Y_test, distilled_prediction)
    mi_teacher_distilled = mutual_info_score(teacher_prediction, distilled_prediction)
    mi_student_distilled = mutual_info_score(student_prediction, distilled_prediction)
    mi_teacher_student = mutual_info_score(teacher_prediction, student_prediction)

    # compute our mutual information with L/C*log(L/C)
    L_student = X_train.shape[1] # number of literals for the student
    L_teacher = X_train.shape[1] # number of literals for the teacher
    L_distilled = X_train_downsampled.shape[1] # number of literals for the distilled
    C_student = params['student_num_clauses'] # number of clauses for the student
    C_teacher = params['teacher_num_clauses'] # number of clauses for the teacher
    C_distilled = params['student_num_clauses'] # number of clauses for the distilled
    
    info_teacher = L_teacher/C_teacher*np.log(L_teacher/C_teacher)
    info_student = L_student/C_student*np.log(L_student/C_student)
    info_distilled = L_distilled/C_distilled*np.log(L_distilled/C_distilled)

    print(f"Information (teacher) (L_T/C_T*log(L_T/C_T)): {info_teacher:.4f}")
    print(f"Information (student) (L_S/C_S*log(L_S/C_S)): {info_student:.4f}")
    print(f"Information (distilled) (L_D/C_D*log(L_D/C_D)): {info_distilled:.4f}")

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
            "avg_time_train_teacher": results[TIME_TRAIN_TEACHER].mean(),
            "avg_time_train_student": results[TIME_TRAIN_STUDENT].mean(),
            "avg_time_train_distilled": results[TIME_TRAIN_DISTILLED].mean(),
            "avg_time_test_teacher": results[TIME_TEST_TEACHER].mean(),
            "avg_time_test_student": results[TIME_TEST_STUDENT].mean(),
            "avg_time_test_distilled": results[TIME_TEST_DISTILLED].mean(),
            "total_time": total_time,
            "num_clauses_dropped": num_clauses_dropped,
            "num_clauses_dropped_percentage": reduction_percentage
        },
        "mutual_information": {
            "info_teacher": info_teacher, # calculated using L/C*log(L/C)
            "info_student": info_student, # calculated using L/C*log(L/C)
            "info_distilled": info_distilled, # calculated using L/C*log(L/C)
            "mi_true_student": mi_true_student, # calculated using sklearn
            "mi_true_teacher": mi_true_teacher, # calculated using sklearn
            "mi_true_distilled": mi_true_distilled, # calculated using sklearn
            "mi_teacher_distilled": mi_teacher_distilled, # calculated using sklearn
            "mi_student_distilled": mi_student_distilled, # calculated using sklearn
            "mi_teacher_student": mi_teacher_student # calculated using sklearn
        },
        "helpful_for_calculations":{
            "L_student": L_student,
            "L_teacher": L_teacher,
            "L_distilled": L_distilled,
            "C_student": C_student,
            "C_teacher": C_teacher,
            "C_distilled": C_distilled
        },
        "params": params,
        "experiment_name": experiment_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": experiment_id,
        "results": results.to_dict(),
    }

    # save output
    fpath = os.path.join(folderpath, experiment_id)
    save_json(output, os.path.join(fpath, OUTPUT_JSON_PATH))
    results.to_csv(os.path.join(fpath, RESULTS_CSV_PATH))
    if save_all:
        save_pkl(baseline_teacher_tm, os.path.join(folderpath, experiment_id, TEACHER_BASELINE_MODEL_PATH))
        save_pkl(baseline_student_tm, os.path.join(folderpath, experiment_id, STUDENT_BASELINE_MODEL_PATH))
        save_pkl(distilled_tm, os.path.join(folderpath, experiment_id, DISTILLED_MODEL_PATH))

    # plot results and save
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.plot(results[ACC_TEST_DISTILLED], label="Distilled")
    plt.plot(results[ACC_TEST_TEACHER], label="Teacher", alpha=0.5)
    plt.plot(results[ACC_TEST_STUDENT], label="Student", alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(range(0, len(results), 5))
    plt.legend(loc="upper left")
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(fpath, ACCURACY_PNG_PATH))
    plt.close()

    return output, results

