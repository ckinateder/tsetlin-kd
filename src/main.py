from distillation import distillation_experiment
import os
from datasets import MNISTDataset, MNIST3DDataset, FashionMNISTDataset, KMNISTDataset, IMDBDataset, EMNISTLettersDataset
from util import load_or_create
import numpy as np
import random
# set seeds
np.random.seed(0)
random.seed(0)

"""
So far, these are the best params:

distilled:
    (mnist_dataset, "MNIST", { "teacher_num_clauses": 600, "student_num_clauses": 100, "T": 20, "s": 10.0,"teacher_epochs": 60, "student_epochs": 60 , "temperature": 3.0}, {"overwrite": False}),

    EMNIST:
        (emnist_dataset, "EMNIST", 
            {
                "teacher": { "C": 800, "T": 100, "s": 4.0, "epochs": 40 },
                "student": { "C": 100, "T": 60, "s": 4.0, "epochs": 80 },
                "temperature": 3,
            },
            {"overwrite": True}
        )
        (mnist_dataset, "MNIST", 
            {
                "teacher": { "C": 1000, "T": 10, "s": 4.0, "epochs": 40 },
                "student": { "C": 100, "T": 10, "s": 4.0, "epochs": 80 },
                "temperature": 3.0,
            },
            {"overwrite": True}
        ),
"""

if __name__ == "__main__":
    # load datasets.
    print("Loading datasets...")
    kmnist_dataset = load_or_create(os.path.join("data", "kmnist_dataset.pkl"), KMNISTDataset)
    mnist3d_dataset = load_or_create(os.path.join("data", "mnist3d_dataset.pkl"), MNIST3DDataset)
    mnist_dataset = load_or_create(os.path.join("data", "mnist_dataset.pkl"), MNISTDataset)
    fashion_mnist_dataset = load_or_create(os.path.join("data", "fashion_mnist_dataset.pkl"), FashionMNISTDataset)
    imdb_dataset = load_or_create(os.path.join("data", "imdb_dataset.pkl"), IMDBDataset)
    emnist_dataset = load_or_create(os.path.join("data", "emnist_dataset.pkl"), EMNISTLettersDataset)
    print("Datasets loaded")
        
    #run distilled experiments
    # this goes (dataset, name, params, kwargs)
    one_off_dir = os.path.join("results")
    distilled_experiments = [
        (emnist_dataset, "EMNIST", 
            {
                "teacher": { "C": 800, "T": 100, "s": 4.0, "epochs": 60 },
                "student": { "C": 100, "T": 60, "s": 4.0, "epochs": 120 },
                "temperature": 4,
            },
            {"overwrite": False}
        ),
        (emnist_dataset, "EMNIST", 
            {
                "teacher": { "C": 1000, "T": 200, "s": 5.0, "epochs": 60 },
                "student": { "C": 100, "T": 200, "s": 5.0, "epochs": 120 },
                "temperature": 3,
            },
            {"overwrite": False}
        ),
        (kmnist_dataset, "KMNIST", 
            {
                "teacher": { "C": 1000, "T": 400, "s": 4.0, "epochs": 30 },
                "student": { "C": 100, "T": 400, "s": 4.0, "epochs": 60 },
                "temperature": 3,
            },
            {"overwrite": False}
        ),
        (kmnist_dataset, "KMNIST", 
            {
                "teacher": { "C": 2000, "T": 600, "s": 4.0, "epochs": 60 },
                "student": { "C": 200, "T": 1000, "s": 3.5, "epochs": 120 },
                "temperature": 3,
            },
            {"overwrite": False}
        ),
        (mnist_dataset, "MNIST", 
            {
                "teacher": { "C": 1000, "T": 10, "s": 4.0, "epochs": 40 },
                "student": { "C": 100, "T": 10, "s": 4.0, "epochs": 80 },
                "temperature": 3.0,
            },
            {"overwrite": True}
        ),
        (mnist3d_dataset, "MNIST3D", 
            {
                "teacher": { "C": 2000, "T": 100, "s": 8.0, "epochs": 30 },
                "student": { "C": 200, "T": 100, "s": 8.0, "epochs": 60 },
                "temperature": 4.0,
            },
            {"overwrite": False}
        ),
        (imdb_dataset, "IMDB", 
            {
                "teacher": { "C": 10000, "T": 6000, "s": 4.0, "epochs": 20 },
                "student": { "C": 2000, "T": 6000, "s": 4.0, "epochs": 60 },
                "temperature": 4.0,
            },
            {"overwrite": False}
        ),
    ]
    
    print("Running distilled experiments")
    for dataset, name, params, kwargs in distilled_experiments:
        kwargs["folderpath"] = one_off_dir
        kwargs["save_all"] = True
        distillation_experiment(dataset, name, params, **kwargs)
        
    