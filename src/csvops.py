# csvoperations

import csv
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

def iterate_over_file_in_folder(folder="experiments", file_extension=".json"):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    yield data, file_path

def combine_into_cols():
    # make a single csv with all the results. prepend the experiment name to each result column
    output = pd.DataFrame()
    for experiment, file_path in iterate_over_file_in_folder():
        results = experiment["results"]
        params = experiment["params"]
        experiment_id = experiment["id"]
        experiment_name = experiment["experiment_name"]
        # add columns to the output dataframe
        intermediate = pd.DataFrame(results)
        intermediate.columns = [f"{experiment_name}_{col}" for col in intermediate.columns]
        output = pd.concat([output, intermediate], axis=1)

    # sort columns alphabetically
    output = output.reindex(sorted(output.columns), axis=1)

    output.to_csv(os.path.join("experiments", "one_table.csv"), index=False)

        

def process_mi():
    # make a csv wiht params at the top and experiment id as the row
    mis = pd.DataFrame(columns=["experiment_name", "id", "teacher_num_clauses", "student_num_clauses", "T", "s", "teacher_epochs", "student_epochs", "weighted_clauses","number_of_state_bits", "mutual_info_sd" ,"mutual_info_td"])
    for experiment, file_path in iterate_over_file_in_folder():
        print(file_path)
        params = experiment["params"]
        experiment_id = experiment["id"]

        mis = mis._append({
            "experiment_name": experiment["experiment_name"],
            "id": experiment_id,
            "teacher_num_clauses": params["teacher_num_clauses"],
            "student_num_clauses": params["student_num_clauses"],
            "T": params["T"],
            "s": params["s"],
            "teacher_epochs": params["teacher_epochs"],
            "student_epochs": params["student_epochs"],
            "weighted_clauses": params["weighted_clauses"],
            "number_of_state_bits": params["number_of_state_bits"],
            "mutual_info_sd": experiment["mutual_information"]["sklearn_student"],
            "mutual_info_td": experiment["mutual_information"]["sklearn_teacher"]
        }, ignore_index=True)
    # sort alph
    mis = mis.sort_values(by="experiment_name")
    mis.to_csv(os.path.join("experiments", "mutual_info.csv"), index=False)
    return mis
                    
def add_analyses():
    """
    
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

    """
    import numpy as np
    for experiment, file_path in iterate_over_file_in_folder():
        results = experiment["results"]
        results_pd = pd.DataFrame(results)
        experiment["analysis"] = { 
            "avg_acc_test_teacher": results_pd["acc_test_teacher"].mean(),
            "std_acc_test_teacher": results_pd["acc_test_teacher"].std(),
            "avg_acc_test_student": results_pd["acc_test_student"].mean(),
            "std_acc_test_student": results_pd["acc_test_student"].std(),
            "avg_acc_test_distilled": results_pd["acc_test_distilled"].mean(),
            "std_acc_test_distilled": results_pd["acc_test_distilled"].std(),
            "final_acc_test_distilled": results_pd["acc_test_distilled"].iloc[-1],
            "sum_time_train_teacher": results_pd["time_train_teacher"].sum(),
            "sum_time_train_student": results_pd["time_train_student"].sum(),
            "sum_time_train_distilled": results_pd["time_train_distilled"].sum()
        }
        with open(file_path, 'w') as f:
            json.dump(experiment, f, indent=4)

def make_charts():
    for experiment, file_path in iterate_over_file_in_folder():
        results = experiment["results"]
        results_pd = pd.DataFrame(results)
        params = experiment["params"]
        del params["number_of_state_bits"]
        del params["weighted_clauses"]

        # make a chart of the accuracies
        plt.figure(figsize=(8,6))
        plt.plot(results_pd["acc_test_distilled"], label="Distilled")
        plt.plot(results_pd["acc_test_teacher"], label="Teacher", alpha=0.5)
        plt.plot(results_pd["acc_test_student"], label="Student", alpha=0.5)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(experiment["experiment_name"])
        plt.xticks(range(0, len(results_pd), 5))
        plt.legend(loc="upper left")
        # add text of parameters
        params_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
        plt.gcf().text(0.68, 0.14, params_text, fontsize=8, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=1))
        plt.savefig(file_path.replace(".json", ".png"))
        plt.close()

def make_accuracy_table():
    """
    experiment, avg_acc_test_teacher, std_acc_test_teacher, avg_acc_test_student, std_acc_test_student, avg_acc_test_distilled, std_acc_test_distilled, final_acc_test_distilled, mutual_info_sd, mutual_info_td
    """
    table = pd.DataFrame(columns=["experiment", "avg_acc_test_teacher", "std_acc_test_teacher", "avg_acc_test_student", "std_acc_test_student", "avg_acc_test_distilled", "std_acc_test_distilled", "final_acc_test_distilled", "mutual_info_sd", "mutual_info_td"])
    for experiment, file_path in iterate_over_file_in_folder():
        results = experiment["results"]
        results_pd = pd.DataFrame(results)
        params = experiment["params"]
        new_row = pd.DataFrame([{
            "experiment": experiment["experiment_name"],
            "avg_acc_test_teacher": round(results_pd["acc_test_teacher"].mean(), 3),
            "std_acc_test_teacher": round(results_pd["acc_test_teacher"].std(), 3),
            "avg_acc_test_student": round(results_pd["acc_test_student"].mean(), 3),
            "std_acc_test_student": round(results_pd["acc_test_student"].std(), 3),
            "avg_acc_test_distilled": round(results_pd["acc_test_distilled"].mean(), 3),
            "std_acc_test_distilled": round(results_pd["acc_test_distilled"].std(), 3),
            "final_acc_test_distilled": round(results_pd["acc_test_distilled"].iloc[-1], 3),
            "mutual_info_sd": round(experiment["mutual_information"]["sklearn_student"], 3),
            "mutual_info_td": round(experiment["mutual_information"]["sklearn_teacher"], 3)
        }])
        table = pd.concat([table, new_row], ignore_index=True)
        
    table = table.sort_values(by="experiment")
    table.to_csv(os.path.join("experiments", "accuracy_table.csv"), index=False)
        

if __name__ == "__main__":
    #combine_into_cols()
    make_accuracy_table()