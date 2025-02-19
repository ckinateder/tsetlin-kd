from distillation import distillation_experiment
from downsample import downsample_experiment
import os
from datasets import MNISTDataset, MNIST3DDataset, FashionMNISTDataset, KMNISTDataset, IMDBDataset
from util import load_or_create
import numpy as np
import random
# set seeds
np.random.seed(0)
random.seed(0)

"""
So far, these are the best params:

distilled:
    (kmnist_dataset, "KMNIST", { "teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 600, "s": 5, "teacher_epochs": 20, "student_epochs": 60, "downsample": 0.02 }, {"overwrite": False}),
    (mnist3d_dataset, "MNIST3D", { "teacher_num_clauses": 1500, "student_num_clauses": 50, "T": 100, "s": 3.0, "teacher_epochs": 20, "student_epochs": 70 , "downsample": 0.0}, {"overwrite": False})
    (mnist_dataset, "MNIST", { "teacher_num_clauses": 2000, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 80, "downsample": 0.02 }, {"overwrite": False}),
    (fashion_mnist_dataset, "FashionMNIST", { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 60, "s": 20.0,"teacher_epochs": 20, "student_epochs": 80, "downsample": 0 }, {"overwrite": False}),
    (imdb_dataset, "IMDB", { "teacher_num_clauses": 10000, "student_num_clauses": 2000, "T": 6000, "s": 4.0, "teacher_epochs": 30, "student_epochs": 60, "downsample": 0 }, {"overwrite": False}),

downsample:
    (mnist3d_dataset, "MNIST3D-Downsample", { "teacher_num_clauses": 1500, "student_num_clauses": 50, "T": 100, "s": 3.0, "teacher_epochs": 20, "student_epochs": 70 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], {"overwrite": False}),
    (-=-, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], {"overwrite": False}),
    (mnist_dataset, "MNIST-Downsample", {"teacher_num_clauses": 1200, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 80 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35], {"overwrite": False}),
    (kmnist_dataset, "KMNIST-Downsample", {"teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 600, "s": 5, "teacher_epochs": 20, "student_epochs": 60 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], {"overwrite": False}),
    (imdb_dataset, "IMDB-Downsample", {"teacher_num_clauses": 10000, "student_num_clauses": 2000, "T": 6000, "s": 4.0, "teacher_epochs": 30, "student_epochs": 90 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], {"overwrite": False}),

"""

if __name__ == "__main__":
    # load datasets.
    print("Loading datasets...")
    kmnist_dataset = load_or_create(os.path.join("data", "kmnist_dataset.pkl"), KMNISTDataset)
    mnist3d_dataset = load_or_create(os.path.join("data", "mnist3d_dataset.pkl"), MNIST3DDataset)
    mnist_dataset = load_or_create(os.path.join("data", "mnist_dataset.pkl"), MNISTDataset)
    fashion_mnist_dataset = load_or_create(os.path.join("data", "fashion_mnist_dataset.pkl"), FashionMNISTDataset)
    imdb_dataset = load_or_create(os.path.join("data", "imdb_dataset.pkl"), IMDBDataset)
    print("Datasets loaded")
    
    # run distilled experiments
    # this goes (dataset, name, params, kwargs)
    one_off_dir = os.path.join("results", "top_singles")
    distilled_experiments = [
        # done (mnist3d_dataset, "MNIST3D", { "teacher_num_clauses": 2000, "student_num_clauses": 200, "T": 100, "s": 8.0, "teacher_epochs": 30, "student_epochs": 60, "downsample": 0 }, {"overwrite": False}),
        # (fashion_mnist_dataset, "FashionMNIST", { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 60, "s": 20.0,"teacher_epochs": 20, "student_epochs": 80, "downsample": 0 }, {"overwrite": False}),
        # done (kmnist_dataset, "KMNIST", { "teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 600, "s": 5, "teacher_epochs": 20, "student_epochs": 60, "downsample": 0.02 }, {"overwrite": False}),
        # done (mnist_dataset, "MNIST", { "teacher_num_clauses": 2000, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 80, "downsample": 0.02 }, {"overwrite": False}),
        #(imdb_dataset, "IMDB", { "teacher_num_clauses": 10000, "student_num_clauses": 2000, "T": 80*100, "s": 10.0, "teacher_epochs": 30, "student_epochs": 90, "downsample": 0.01 }, {"overwrite": False}),
    ]
    
    print("Running distilled experiments")
    for dataset, name, params, kwargs in distilled_experiments:
        kwargs["folderpath"] = one_off_dir
        kwargs["save_all"] = True
        distillation_experiment(dataset, name, params, **kwargs)
    
    # run downsample experiments
    # this goes (dataset, name, params, downsamples, kwargs)
    downsample_dir = os.path.join("results", "downsample")
    downsample_experiments = [        
        # (fashion_mnist_dataset, "FashionMNIST-Downsample", { "teacher_num_clauses": 800, "student_num_clauses": 150, "T": 60, "s": 20.0,"teacher_epochs": 20, "student_epochs": 80 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], {"overwrite": False}),
        #(mnist_dataset, "MNIST-Downsample-1200", { "teacher_num_clauses": 1200, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 80 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40], {"overwrite": False}),
        #(mnist3d_dataset, "MNIST3D-Downsample", { "teacher_num_clauses": 1500, "student_num_clauses": 50, "T": 100, "s": 3.0, "teacher_epochs": 20, "student_epochs": 70 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], {"overwrite": False}),
        #(kmnist_dataset, "KMNIST-Downsample", {"teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 600, "s": 5, "teacher_epochs": 20, "student_epochs": 60 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], {"overwrite": False}),
        #(imdb_dataset, "IMDB-Downsample", {"teacher_num_clauses": 10000, "student_num_clauses": 2000, "T": 6000, "s": 4.0, "teacher_epochs": 30, "student_epochs": 90 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], {"overwrite": False}),
        (imdb_dataset, "IMDB-Downsample-Take-2", {"teacher_num_clauses": 10000, "student_num_clauses": 2000, "T": 6000, "s": 4.0, "teacher_epochs": 30, "student_epochs": 90 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35], {"overwrite": False}),
    ]
    
    print("Running downsample experiments")
    for dataset, name, params, downsamples, kwargs in downsample_experiments:
        kwargs["folderpath"] = downsample_dir
        kwargs["save_all"] = True
        downsample_experiment(dataset, name, params, downsamples, **kwargs)
    
