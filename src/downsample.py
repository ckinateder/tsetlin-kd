from distillation import distillation_experiment, validate_params
from util import load_pkl, load_json, save_json, make_dir
from datasets import Dataset
from __init__ import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def downsample_experiment(
    dataset: Dataset,
    experiment_name,
    params=DISTILLED_DEFAULTS,
    downsamples=DOWNSAMPLE_DEFAULTS,
    folderpath=DEFAULT_FOLDERPATH,
    overwrite: bool = True,
    save_all: bool = False
) -> dict:
    """
    Run a downsample experiment comparing teacher, student, and distilled models.
    Note that the first downsample is the original data, so it's not included in the downsamples list.
    0 will be prepended to the downsamples list to ensure that the original data is used as a baseline.
    Duplicate downsamples are removed and the list is sorted.

    Args:
        dataset (Dataset): The dataset to use for the experiment
        experiment_name (str): Name of the experiment
        params (dict, optional): Parameters for the experiment. Defaults to DOWNSAMPLE_DEFAULTS.
        downsamples (list, optional): List of downsample rates to use for the experiment. Defaults to DOWNSAMPLE_DEFAULTS.
        folderpath (str, optional): Path to save experiment results. Defaults to DEFAULT_FOLDERPATH.
        overwrite (bool, optional): Whether to overwrite existing experiment results. Defaults to True.
        save_all (bool, optional): Whether to save all models. Defaults to True.
    Returns:
        dict: Dictionary containing experiment results including:
            - Teacher, student and distilled model accuracies
            - Training and testing times
            - Number of clauses dropped during distillation
            - Total experiment time
    """
    downsamples = [0] + downsamples # add 0 to the downsamples
    downsamples = sorted(list(set(downsamples))) # remove duplicates and sort
    print(f"Running Downsample Experiment {experiment_name} with params: {params} and downsamples: {downsamples}")
    
    # train distilled model first, fully raw, NO baseline models and NO downsampling
    params["downsample"] = 0
    print("Training distilled model first, fully raw, NO baseline models and NO downsampling")
    subfolderpath = os.path.join(folderpath, experiment_name)

    # check if the experiment already exists
    expected_id = validate_params(params, "ds")
    if not overwrite and os.path.exists(os.path.join(subfolderpath, expected_id, "output.json")) and os.path.exists(os.path.join(subfolderpath, expected_id, "results.csv")):
        print(f"Experiment {expected_id} already exists, loading results")
        original_output = load_json(os.path.join(subfolderpath, expected_id, "output.json"))
        original_results_pd = pd.read_csv(os.path.join(subfolderpath, expected_id, "results.csv"), index_col=0)
        original_id = expected_id
    else:
        make_dir(subfolderpath, overwrite=overwrite)
        original_output, original_results_pd = distillation_experiment(dataset, "ds", params, subfolderpath, save_all=True)
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
    failed_downsamples = []
    for i, downsample in enumerate(downsamples[1:]): # skip the first one because it's the original
        print(f"Training distilled model with downsampling {downsample}")
        params["downsample"] = downsample
        output_dict, _ = distillation_experiment(dataset, "ds", params, subfolderpath, 
                                               baseline_teacher_model=teacher_model, 
                                               baseline_student_model=student_model, 
                                               pretrained_teacher_model=pretrained_teacher_model, 
                                               prefilled_results=original_results_pd,
                                               overwrite=overwrite,
                                               save_all=save_all)
        all_outputs.append(output_dict)
        if output_dict is None:
            print(f"Skipping downsample {downsample} because it failed")
            failed_downsamples.append(i)
    
    if len(failed_downsamples) > 0:
        print(f"Failed to train distilled models with downsamples: {[downsamples[i] for i in failed_downsamples]}")
        # remove failed downsamples from downsamples list
        downsamples = [d for i, d in enumerate(downsamples) if i not in failed_downsamples]
        all_outputs = [o for i, o in enumerate(all_outputs) if i not in failed_downsamples]
    
    downsamples = np.array(downsamples)
    print("Done training distilled models with downsampling")

    # get original accuracy
    original_final_acc = original_output["analysis"]["final_acc_test_distilled"]
    original_avg_acc = original_output["analysis"]["avg_acc_test_distilled"]
    original_total_training_time = original_output["analysis"]["sum_time_train_distilled"]
    original_avg_training_time = original_output["analysis"]["avg_time_train_distilled"]
    original_i_distilled = original_output["mutual_information"]["I_distilled"]
    original_reduction_percentage = 0
    original_avg_test_time = original_output["analysis"]["avg_time_test_distilled"]

    # now plot the results. Y value is average accuracy of distilled model plotted over different downsamples
    # add a line for baseline teacher and baseline student
    baseline_student_avg_acc = original_output["analysis"]["avg_acc_test_student"]
    baseline_teacher_avg_acc = original_output["analysis"]["avg_acc_test_teacher"]
    baseline_student_final_acc = original_output["analysis"]["final_acc_test_student"]
    baseline_teacher_final_acc = original_output["analysis"]["final_acc_test_teacher"]
    baseline_student_total_training_time = original_output["analysis"]["sum_time_train_student"]
    baseline_teacher_total_training_time = original_output["analysis"]["sum_time_train_teacher"]
    baseline_student_avg_training_time = original_output["analysis"]["avg_time_train_student"]
    baseline_teacher_avg_training_time = original_output["analysis"]["avg_time_train_teacher"]
    baseline_i_student = original_output["mutual_information"]["I_student"]
    baseline_i_teacher = original_output["mutual_information"]["I_teacher"]
    baseline_teacher_avg_test_time = original_output["analysis"]["avg_time_test_teacher"]
    baseline_student_avg_test_time = original_output["analysis"]["avg_time_test_student"]

    # get final and average distilled accuracy
    all_final_acc = np.array([original_final_acc] + [output["analysis"]["final_acc_test_distilled"] for output in all_outputs])
    all_avg_acc = np.array([original_avg_acc] + [output["analysis"]["avg_acc_test_distilled"] for output in all_outputs])
    all_total_training_time = np.array([original_total_training_time] + [output["analysis"]["sum_time_train_distilled"] for output in all_outputs])
    all_avg_training_time = np.array([original_avg_training_time] + [output["analysis"]["avg_time_train_distilled"] for output in all_outputs])
    all_reduction_percentage = np.array([original_reduction_percentage] + [output["analysis"]["num_clauses_dropped_percentage"] for output in all_outputs])
    all_avg_test_time = np.array([original_avg_test_time] + [output["analysis"]["avg_time_test_distilled"] for output in all_outputs])
    
    # get log information
    all_i_distilled = np.array([original_i_distilled] + [output["mutual_information"]["I_distilled"] for output in all_outputs])

    # put into dataframe
    all_results = pd.DataFrame({
        "downsample": downsamples,
        "final_acc": all_final_acc,
        "avg_acc": all_avg_acc,
        "total_training_time": all_total_training_time,
        "avg_training_time": all_avg_training_time,
        "I_distilled": all_i_distilled
    })
    all_results.to_csv(os.path.join(subfolderpath, "downsample_results.csv"))

    ## plots
    horiz_alpha = 0.8
    marker_size = 3
    x_ticks = np.arange(0, downsamples.max()+0.05, 0.05)

    # plot final accuracy
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.axhline(y=baseline_teacher_final_acc, linestyle=':', color="orange", alpha=horiz_alpha, label="Teacher")
    plt.axhline(y=baseline_student_final_acc, linestyle=':', color="green", alpha=horiz_alpha, label="Student")
    plt.plot(downsamples, all_final_acc, marker='o', markersize=marker_size, label="Distilled")
    plt.xlabel("Downsample Rate")
    plt.ylabel("Final Accuracy (%)")
    plt.legend(loc="upper right")
    yticks = plt.yticks()[0]
    if yticks.shape[0] <= PLOT_FIGSIZE[1]+2:
        plt.yticks(np.arange(yticks.min(), plt.yticks()[0].max(), (yticks[1]-yticks[0])/2))
    plt.xticks(x_ticks)
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(subfolderpath, "downsample_results_final_acc.png"))
    plt.close()

    # plot average accuracy
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.axhline(y=baseline_teacher_avg_acc, linestyle=':', color="orange", alpha=horiz_alpha, label="Teacher")
    plt.axhline(y=baseline_student_avg_acc, linestyle=':', color="green", alpha=horiz_alpha, label="Student")
    plt.plot(downsamples, all_avg_acc, marker='o', markersize=marker_size, label="Distilled")
    plt.xlabel("Downsample Rate")
    plt.ylabel("Average Accuracy (%)")
    plt.legend(loc="upper right")
    yticks = plt.yticks()[0]
    if yticks.shape[0] <= PLOT_FIGSIZE[1]+2:
        plt.yticks(np.arange(yticks.min(), plt.yticks()[0].max(), (yticks[1]-yticks[0])/2))
    plt.xticks(x_ticks)
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(subfolderpath, "downsample_results_avg_acc.png"))
    plt.close()

    # plot total training time
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.axhline(y=baseline_teacher_total_training_time, linestyle=':', color="orange", alpha=horiz_alpha, label="Teacher")
    plt.axhline(y=baseline_student_total_training_time, linestyle=':', color="green", alpha=horiz_alpha, label="Student")
    plt.plot(downsamples, all_total_training_time, marker='o', markersize=marker_size, label="Distilled")
    plt.xlabel("Downsample Rate")
    plt.ylabel("Total Training Time (s)")
    plt.legend(loc="upper right")
    yticks = plt.yticks()[0]
    if yticks.shape[0] <= PLOT_FIGSIZE[1]+2:
        plt.yticks(np.arange(yticks.min(), plt.yticks()[0].max(), (yticks[1]-yticks[0])/2))
    plt.xticks(x_ticks)
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(subfolderpath, "downsample_results_total_training_time.png"))
    plt.close()

    # plot average training time
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.axhline(y=baseline_teacher_avg_training_time, linestyle=':', color="orange", alpha=horiz_alpha, label="Teacher")
    plt.axhline(y=baseline_student_avg_training_time, linestyle=':', color="green", alpha=horiz_alpha, label="Student")
    plt.plot(downsamples, all_avg_training_time, marker='o', markersize=marker_size, label="Distilled")
    plt.xlabel("Downsample Rate")
    plt.ylabel("Epoch Training Time (s)")
    plt.legend(loc="upper right")
    yticks = plt.yticks()[0]
    if yticks.shape[0] <= PLOT_FIGSIZE[1]+2:
        plt.yticks(np.arange(yticks.min(), plt.yticks()[0].max(), (yticks[1]-yticks[0])/2))
    plt.xticks(x_ticks)
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(subfolderpath, "downsample_results_avg_training_time.png"))
    plt.close()

    # plot average test time
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.axhline(y=baseline_teacher_avg_test_time, linestyle=':', color="orange", alpha=horiz_alpha, label="Teacher")
    plt.axhline(y=baseline_student_avg_test_time, linestyle=':', color="green", alpha=horiz_alpha, label="Student")
    plt.plot(downsamples, all_avg_test_time, marker='o', markersize=marker_size, label="Distilled")
    plt.xlabel("Downsample Rate")
    plt.ylabel("Average Test Time (s)")
    plt.legend(loc="upper right")
    yticks = plt.yticks()[0]
    if yticks.shape[0] <= PLOT_FIGSIZE[1]+2:
        plt.yticks(np.arange(yticks.min(), plt.yticks()[0].max(), (yticks[1]-yticks[0])/2))
    plt.xticks(x_ticks)
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(subfolderpath, "downsample_results_avg_test_time.png"))
    plt.close()

    # plot reduction percentage
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.plot(downsamples, all_reduction_percentage, marker='o', markersize=marker_size, label="Reduction")
    plt.xlabel("Downsample Rate")
    plt.ylabel("Clause Reduction Percentage (%)")
    plt.legend(loc="upper left")
    plt.yticks(np.arange(0, 100, 10))
    plt.xticks(x_ticks)
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(subfolderpath, "downsample_results_reduction_percentage.png"))
    plt.close()

    # plot information
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.axhline(y=baseline_i_teacher, linestyle=':', color="orange", alpha=horiz_alpha, label="Teacher")
    plt.axhline(y=baseline_i_student, linestyle=':', color="green", alpha=horiz_alpha, label="Student")
    plt.plot(downsamples, all_i_distilled, marker='o', markersize=marker_size, label="Distilled")
    plt.xlabel("Downsample Rate")
    plt.ylabel("Information (nats)")
    plt.legend(loc="upper right")
    plt.xticks(x_ticks)
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(subfolderpath, "downsample_results_information.png"))
    plt.close()

    # save params
    params_path = os.path.join(subfolderpath, "params.json")
    save_json(params, params_path)

    return all_outputs