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
    for experiment in iterate_over_file_in_folder():
        results = experiment["results"]
        params = experiment["params"]
        experiment_id = experiment["id"]

        

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
                    
if __name__ == "__main__":
    process_mi()