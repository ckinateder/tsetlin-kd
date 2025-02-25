#from stats import entropy, normalize, softmax, joint_probs, mutual_information, kl_divergence
import numpy as np
from time import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from util import save_json, load_pkl, save_pkl, make_dir, rm_file, load_json
from datetime import datetime
from tqdm import tqdm
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from datasets import Dataset
from __init__ import *
from prodict import Prodict

def train_step(model, X_train, Y_train, X_test, Y_test, epoch: int, soft_labels: np.ndarray | None = None) -> tuple[float, float, float]:
    """
    Train a model for one epoch and return test accuracy and timing information.

    Args:
        model: The TsetlinMachine model to train
        X_train: Training data features
        Y_train: Training data labels 
        X_test: Test data features
        Y_test: Test data labels
        soft_labels: Soft labels for the training data
    Returns:
        tuple[float, float, float]: Test accuracy percentage, training time, and testing time
    """
    tqdm.write(f'Epoch {epoch:>3}:', end=' ')

    start_training = time()
    if soft_labels is None:
        model.fit(X_train, Y_train, epochs=1, incremental=True)
    else:
        model.fit_soft(X_train, Y_train, epochs=1, incremental=True, soft_labels=soft_labels)
    stop_training = time()
    tqdm.write(f'Training time: {stop_training-start_training:.2f} s', end=' ')
    start_testing = time()
    prediction = model.predict(X_test)
    result = 100*(prediction == Y_test).mean()
    stop_testing = time()
    tqdm.write(f'Testing time: {stop_testing-start_testing:.2f} s, Test accuracy: {result:.2f}%')
    return result, stop_training-start_training, stop_testing-start_testing

def validate_params(params: dict, experiment_name: str) -> str:
    """
    Validate the parameters for the experiment.
    Example valid params:
        {
            "teacher": {
                "C": 1000,
                "T": 10,
                "s": 5,
                "epochs": 30,
            },
            "student": {
                "C": 100,
                "T": 10,
                "s": 5,
                "epochs": 60,
            },
            "temperature": 4.0,
            "weighted_clauses": True,
            "number_of_state_bits": 8,
        }

    Args:
        params (dict): Parameters for the experiment
        experiment_name (str): Name of the experiment

    Returns:
        str: id of the experiment
    """
    # check that the parameters are valid
    assert "teacher" in params, "teacher parameters are required"
    assert "student" in params, "student parameters are required"
    assert all(key in params["teacher"] for key in ["C", "T", "s", "epochs"]), "teacher parameters are required"
    assert all(key in params["student"] for key in ["C", "T", "s", "epochs"]), "student parameters are required"
    assert "temperature" in params, "temperature is required"
    assert "weighted_clauses" in params, "weighted_clauses is required"
    assert "number_of_state_bits" in params, "number_of_state_bits is required"

    params["combined_epochs"] = params["teacher"]["epochs"] + params["student"]["epochs"]
    
    # generate experiment id
    exid = f"{experiment_name.replace(' ', '-')}_tC{params['teacher']['C']}_sC{params['student']['C']}_" \
           f"tT{params['teacher']['T']}_sT{params['student']['T']}_ts{params['teacher']['s']}_ss{params['student']['s']}_" \
           f"te{params['teacher']['epochs']}_se{params['student']['epochs']}"    
    return exid


def distillation_experiment(
    dataset: Dataset,
    experiment_name: str,
    params: dict = DISTILLED_DEFAULTS_NESTED,
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
    print(f"Starting experiment {experiment_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # check that the data is valid
    dataset.validate_lengths()
    X_train, Y_train, X_test, Y_test = dataset.get_data()

    # fill in missing parameters with defaults
    for key, value in DISTILLED_DEFAULTS_NESTED.items():
        if key not in params:
            print(f"Parameter {key} not specified, using default value {value}")
            params[key] = value

    print(f"Using params: {params}")

    # get experiment id
    experiment_id = validate_params(params, experiment_name)
    print(f"Experiment ID: {experiment_id}")

    params = Prodict.from_dict(params)
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
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    baseline_teacher_tm = MultiClassTsetlinMachine(
        params.teacher.C, params.teacher.T, params.teacher.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    distilled_tm = MultiClassTsetlinMachine(
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)

    # create a results dataframe
    if prefilled_results is None: 
        results = pd.DataFrame(columns=RESULTS_COLUMNS, index=range(params.combined_epochs))
    else:
        results = prefilled_results
        assert results.columns.tolist() == RESULTS_COLUMNS, \
            f"Prefilled results columns do not match expected columns: {results.columns.tolist()} != {RESULTS_COLUMNS}"
        assert results.index.to_list() == list(range(params.combined_epochs)), \
            f"Prefilled results index does not match expected index: {results.index.to_list()} != {range(params.combined_epochs)}"

        # make sure it has no nan values
        assert not results.isna().any().any(), "Prefilled results has nan values"

        # delete entries after params['teacher_epochs'] for 'acc_test_distilled', 'time_train_distilled', 'time_test_distilled' columns
        results.loc[params.teacher.epochs:, ACC_TEST_DISTILLED] = np.nan
        results.loc[params.teacher.epochs:, TIME_TRAIN_DISTILLED] = np.nan 
        results.loc[params.teacher.epochs:, TIME_TEST_DISTILLED] = np.nan
        print(f"Prefilled results loaded")

    # train baselines
    # train baseline teacher
    if isinstance(baseline_teacher_model, MultiClassTsetlinMachine):
        print(f"Loading pretrained baseline teacher model")
        baseline_teacher_tm = baseline_teacher_model
    else:
        print(f"Creating a baseline teacher with {params.teacher.C} clauses and training on original data")
        start = time()
        bt_pbar = tqdm(range(params.combined_epochs), desc="Teacher", leave=False, dynamic_ncols=True)
        best_acc = 0
        best_acc_epoch = 0
        for i in bt_pbar:
            result, train_time, test_time = train_step(baseline_teacher_tm, X_train, Y_train, X_test, Y_test, i)
            results.loc[i, ACC_TEST_TEACHER], results.loc[i, TIME_TRAIN_TEACHER], results.loc[i, TIME_TEST_TEACHER] = result, train_time, test_time
            bt_pbar.set_description(f"Teacher: {results[ACC_TEST_TEACHER].mean():.2f}%")

            if i <= params.teacher.epochs - 1 and result > best_acc:
                save_pkl(baseline_teacher_tm, teacher_model_path)
                best_acc = result
                best_acc_epoch = i

            if i == params.teacher.epochs - 1:
                tqdm.write(f"Saved teacher model to {teacher_model_path} @ epoch {best_acc_epoch} (best acc: {best_acc:.2f}%)")

        bt_pbar.close()
        end = time()
        print(f'Baseline teacher training time: {end-start:.2f} s')

    # copy first teacher_epochs results to distilled results
    results.loc[:params.teacher.epochs, ACC_TEST_DISTILLED] = results.loc[:params.teacher.epochs, ACC_TEST_TEACHER]
    results.loc[:params.teacher.epochs, TIME_TRAIN_DISTILLED] = results.loc[:params.teacher.epochs, TIME_TRAIN_TEACHER]
    results.loc[:params.teacher.epochs, TIME_TEST_DISTILLED] = results.loc[:params.teacher.epochs, TIME_TEST_TEACHER]

    # train baseline student
    if isinstance(baseline_student_model, MultiClassTsetlinMachine):
        print(f"Loading pretrained baseline student model")
        baseline_student_tm = baseline_student_model
    else:
        print(f"Creating a baseline student with {params.student.C} clauses and training on original data")
        start = time()
        bs_pbar = tqdm(range(params.combined_epochs), desc="Student", leave=False, dynamic_ncols=True)
        for i in bs_pbar:
            result, train_time, test_time = train_step(baseline_student_tm, X_train, Y_train, X_test, Y_test, i)
            results.loc[i, ACC_TEST_STUDENT], results.loc[i, TIME_TRAIN_STUDENT], results.loc[i, TIME_TEST_STUDENT] = result, train_time, test_time
            bs_pbar.set_description(f"Student: {results[ACC_TEST_STUDENT].mean():.2f}%")

        bs_pbar.close()
        end = time()
        print(f'Baseline student training time: {end-start:.2f} s')

    # train distilled model
    if isinstance(pretrained_teacher_model, MultiClassTsetlinMachine):
        print(f"Loading pretrained teacher model")
        teacher_tm = pretrained_teacher_model
    else:
        print(f"Loading teacher model from {teacher_model_path}, trained for {params.teacher.epochs} epochs")
        teacher_tm = load_pkl(teacher_model_path)
        if not save_all:
            rm_file(teacher_model_path) # remove the teacher model file. we don't need it anymore

    # GET soft labels
    print(f"Initializing student with {params.student.C} clauses from teacher")
    distilled_tm.init_from_teacher(teacher_tm, params.student.C, X_train, Y_train)
    print(f"Generating soft labels from teacher with temperature {params.temperature}")
    soft_labels = teacher_tm.get_soft_labels(X_train, temperature=params.temperature)
    print(f"Soft labels generated")

    start = time()
    print(f"Training distilled model for {params.student.epochs} epochs")
    dt_pbar = tqdm(range(params.teacher.epochs, params.combined_epochs), desc="Distilled", leave=False, dynamic_ncols=True)
    for i in dt_pbar:
        result, train_time, test_time = train_step(distilled_tm, X_train, Y_train, X_test, Y_test, i, soft_labels)
        results.loc[i, ACC_TEST_DISTILLED], results.loc[i, TIME_TRAIN_DISTILLED], results.loc[i, TIME_TEST_DISTILLED] = result, train_time, test_time
        dt_pbar.set_description(f"Distilled: {results[ACC_TEST_DISTILLED].mean():.2f}%")

    dt_pbar.close()
    end = time()

    print(f'Teacher-student training time: {end-start:.2f} s')

    if save_all:
        save_pkl(baseline_teacher_tm, os.path.join(folderpath, experiment_id, TEACHER_BASELINE_MODEL_PATH))
        save_pkl(baseline_student_tm, os.path.join(folderpath, experiment_id, STUDENT_BASELINE_MODEL_PATH))
        save_pkl(distilled_tm, os.path.join(folderpath, experiment_id, DISTILLED_MODEL_PATH))

    total_time = time() - exp_start

    # THIS IS DONE BECAUSE the teacher model will skew inference time when it doesn't actually affect reality
    post_teacher_results = results.iloc[params.teacher.epochs:]

    output = {
        "analysis": {
            "avg_acc_test_teacher": results[ACC_TEST_TEACHER].mean(),
            "avg_acc_test_student": results[ACC_TEST_STUDENT].mean(),
            "avg_acc_test_distilled": post_teacher_results[ACC_TEST_DISTILLED].mean(),

            "std_acc_test_teacher": results[ACC_TEST_TEACHER].std(),
            "std_acc_test_student": results[ACC_TEST_STUDENT].std(),
            "std_acc_test_distilled": post_teacher_results[ACC_TEST_DISTILLED].std(),

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
            "avg_time_train_distilled": post_teacher_results[TIME_TRAIN_DISTILLED].mean(),

            "avg_time_test_teacher": results[TIME_TEST_TEACHER].mean(),
            "avg_time_test_student": results[TIME_TEST_STUDENT].mean(),
            "avg_time_test_distilled": post_teacher_results[TIME_TEST_DISTILLED].mean(),

            "total_time": total_time,
        },
        "params": params.to_dict(),
        "experiment_name": experiment_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": experiment_id,
        "results": results.to_dict(),
    }

    # save output
    fpath = os.path.join(folderpath, experiment_id)
    save_json(output, os.path.join(fpath, OUTPUT_JSON_PATH))
    results.to_csv(os.path.join(fpath, RESULTS_CSV_PATH))

    # plot results and save
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.plot(results[ACC_TEST_DISTILLED], label="Distilled")
    plt.plot(results[ACC_TEST_TEACHER], label="Teacher", alpha=0.5)
    plt.plot(results[ACC_TEST_STUDENT], label="Student", alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    if len(results) < 100:
        plt.xticks(range(0, len(results), 5))
    else:
        plt.xticks(range(0, len(results), 10))
    plt.legend(loc="upper left")
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(fpath, ACCURACY_PNG_PATH))
    plt.close()


    # plot bar chart of test time for each model
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    labels = ["Teacher", "Student", "Distilled"]
    data = [output["analysis"]["avg_time_test_teacher"], output["analysis"]["avg_time_test_student"], output["analysis"]["avg_time_test_distilled"]]
    colors = ["orange", "green", "blue"]
    plt.bar(labels, data, color=colors)
    # add text on top of each bar
    offset = max(data) * 0.02
    for i, label in enumerate(labels):
        plt.text(i, data[i]+offset, f"{data[i]:.3f} s", ha="center", va="bottom")
    
    # take default y ticks and set 10% higher
    yticks = plt.yticks()[0]
    plt.yticks(np.arange(0, yticks.max()*1.1, yticks[1]-yticks[0]))

    #plt.xlabel("Model")
    plt.ylabel("Inference Time (s)")
    plt.savefig(os.path.join(fpath, TEST_TIME_PNG_PATH))
    plt.close()

    return output, results

