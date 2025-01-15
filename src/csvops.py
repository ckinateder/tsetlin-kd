# csvoperations

import csv
import pandas as pd
import json
import os

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
            "sum_time_train_teacher": results_pd["time_train_teacher"].sum(),
            "sum_time_train_student": results_pd["time_train_student"].sum(),
            "sum_time_train_distilled": results_pd["time_train_distilled"].sum()
        }
        with open(file_path, 'w') as f:
            json.dump(experiment, f, indent=4)

if __name__ == "__main__":
    #process_mi()
    add_analyses()