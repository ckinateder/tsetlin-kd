import h5py
import numpy as np
from distillation_experiment import distillation_experiment, downsample_experiment
from torchvision import transforms
import os
import random
from datasets import MNISTDataset, MNIST3DDataset, FashionMNISTDataset, KMNISTDataset, IMDBDataset
from util import save_pkl, load_pkl
# set seeds
np.random.seed(0)
random.seed(0)

if __name__ == "__main__":
    # load datasets.
    print("Loading datasets...")
    kmnist_dataset = KMNISTDataset()
    mnist3d_dataset = MNIST3DDataset()
    mnist_dataset = MNISTDataset()
    fashion_mnist_dataset = FashionMNISTDataset()
    # imdb dataset takes a while to load, so we save it to a file
    if not os.path.exists(os.path.join("data", "imdb_dataset.pkl")):
        print("IMDB dataset not found, creating it")
        imdb_dataset = IMDBDataset()
        save_pkl(imdb_dataset, os.path.join("data", "imdb_dataset.pkl"))
    else:
        imdb_dataset = load_pkl(os.path.join("data", "imdb_dataset.pkl"))
    
    print("Datasets loaded")
    
    # run distilled experiments
    # this goes (dataset, name, params, kwargs)
    #"""
    distilled_experiments = [
        #(kmnist_dataset, "KMNIST", { "teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 600, "s": 5, "teacher_epochs": 20, "student_epochs": 60, "downsample": 0.02 }, {"overwrite": False}),
        #(mnist3d_dataset, "MNIST3D", { "teacher_num_clauses": 2000, "student_num_clauses": 300, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 30, "downsample": 0.02 }, {"overwrite": False}),
        #(mnist_dataset, "MNIST", { "teacher_num_clauses": 2000, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 80, "downsample": 0.02 }, {"overwrite": False}),
        #(fashion_mnist_dataset, "FashionMNIST", { "teacher_num_clauses": 2000, "student_num_clauses": 100, "T": 30, "s": 10.0,"teacher_epochs": 20, "student_epochs": 80, "downsample": 0.02 }, {"overwrite": False}),
        #(imdb_dataset, "IMDB", { "teacher_num_clauses": 10000, "student_num_clauses": 2000, "T": 80*100, "s": 10.0, "teacher_epochs": 30, "student_epochs": 60 }, {"overwrite": False}),
    ]
    
    print("Running distilled experiments")
    for dataset, name, params, kwargs in distilled_experiments:
        distillation_experiment(dataset, name, params, **kwargs)
    #"""

    # run downsample experiments
    # this goes (dataset, name, params, downsamples, kwargs)
    downsample_experiments = [
        (fashion_mnist_dataset, "FashionMNIST-Downsample", {"teacher_num_clauses": 2000, "student_num_clauses": 100, "T": 30, "s": 10.0,"teacher_epochs": 20, "student_epochs": 80 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], {"overwrite": False}),
        (mnist3d_dataset, "MNIST3D-Downsample", {"teacher_num_clauses": 2000, "student_num_clauses": 300, "T": 60, "s": 3.0, "teacher_epochs": 15, "student_epochs": 45 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35], {"overwrite": False}),
        (mnist_dataset, "MNIST-Downsample", {"teacher_num_clauses": 1200, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 80 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35], {"overwrite": False}),
        (kmnist_dataset, "KMNIST-Downsample", {"teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 600, "s": 5, "teacher_epochs": 20, "student_epochs": 60 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], {"overwrite": False}),
        (imdb_dataset, "IMDB-Downsample", {"teacher_num_clauses": 10000, "student_num_clauses": 2000, "T": 80*100, "s": 10.0, "teacher_epochs": 30, "student_epochs": 60 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], {"overwrite": False}),
    ]
    
    print("Running downsample experiments")
    for dataset, name, params, downsamples, kwargs in downsample_experiments:
        downsample_experiment(dataset, name, params, downsamples, **kwargs)
    

    ############################
    print("Prematurely exiting because we already have all the data we need")
    exit()

    ############################

    """### Load MNIST data"""

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # Data booleanization
    X_train = np.where(X_train > 75, 1, 0)
    X_test = np.where(X_test > 75, 1, 0)

    # Input data flattening
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    mnist_experiments = [
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 5, "s": 10.0, "teacher_epochs": 30, "student_epochs": 60 },
        { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 8, "s": 7.0,"teacher_epochs": 30, "student_epochs": 60 },
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 8, "s": 7.0, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 8, "s": 7.0,"teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 1600, "student_num_clauses": 400, "T": 8, "s": 7.0, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 2400, "student_num_clauses": 400, "T": 8, "s": 7.0, "teacher_epochs": 30, "student_epochs": 30 },
    ]

    for i, params in enumerate(mnist_experiments):
        mnist_results, df = distillation_experiment(
            X_train, Y_train, X_test, Y_test, f"MNIST", params)
        print(mnist_results)
        
    """### Load MNIST-3D data"""
    with h5py.File(os.path.join("data", "mnist3d.h5"), "r") as hf:
        X_train = hf["X_train"][:]
        Y_train = hf["y_train"][:]    
        X_test = hf["X_test"][:]  
        Y_test = hf["y_test"][:]  

    # Print shapes with labels
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

    # Data booleanization
    X_train = np.where(X_train > 0.3, 1, 0)
    X_test = np.where(X_test > 0.3, 1, 0)

    X_train = X_train.reshape(X_train.shape[0], 16*16*16)
    X_test = X_test.reshape(X_test.shape[0], 16*16*16)

    # so far, best params are: num_clauses=750, threshold=50, specificity=3.0
    #         { "teacher_num_clauses": 1000, "student_num_clauses": 100, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 20 },

    mnist3d_experiments = [
        { "teacher_num_clauses": 1000, "student_num_clauses": 100, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 30, "downsample": 0.02 },
        { "teacher_num_clauses": 2000, "student_num_clauses": 400, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 30, "downsample": 0.02 },
        { "teacher_num_clauses": 1200, "student_num_clauses": 400, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 30 },
    ]

    for i, params in enumerate(mnist3d_experiments):
        mnist3d_results, df = distillation_experiment(
            X_train, Y_train, X_test, Y_test, f"MNIST-3D", params)
        print(mnist3d_results)

    """### Load IMDB data"""
    (X_train, Y_train), (X_test, Y_test) = prepare_imdb_data()
    imdb_experiments = [
        {"teacher_num_clauses": 10000, "student_num_clauses": 2000, "T": 80*100, "s": 10.0, "teacher_epochs": 30, "student_epochs": 60},
    ]
    
    for i, params in enumerate(imdb_experiments):
        imdb_results, df = distillation_experiment(
            X_train, Y_train, X_test, Y_test, f"IMDB", params)
        print(imdb_results)

    """### Load MNIST-3D data"""
    with h5py.File(os.path.join("data", "mnist3d.h5"), "r") as hf:
        X_train = hf["X_train"][:]
        Y_train = hf["y_train"][:]    
        X_test = hf["X_test"][:]  
        Y_test = hf["y_test"][:]  

    # Print shapes with labels
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

    # Data booleanization
    X_train = np.where(X_train > 0.3, 1, 0)
    X_test = np.where(X_test > 0.3, 1, 0)

    X_train = X_train.reshape(X_train.shape[0], 16*16*16)
    X_test = X_test.reshape(X_test.shape[0], 16*16*16)

    # so far, best params are: num_clauses=750, threshold=50, specificity=3.0
    #         { "teacher_num_clauses": 1000, "student_num_clauses": 100, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 20 },

    mnist3d_experiments = [
        { "teacher_num_clauses": 1000, "student_num_clauses": 100, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 30 },
        { "teacher_num_clauses": 2000, "student_num_clauses": 400, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 30 },
        { "teacher_num_clauses": 1200, "student_num_clauses": 400, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 30 },
    ]

    for i, params in enumerate(mnist3d_experiments):
        mnist3d_results, df = distillation_experiment(
            X_train, Y_train, X_test, Y_test, f"MNIST-3D", params)
        print(mnist3d_results)

    """### Load MNIST data"""

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # Data booleanization
    X_train = np.where(X_train > 75, 1, 0)
    X_test = np.where(X_test > 75, 1, 0)

    # Input data flattening
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    mnist_experiments = [
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 10, "s": 5, "teacher_epochs": 10, "student_epochs": 10 },
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 8, "s": 7.0, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 8, "s": 7.0,"teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 1600, "student_num_clauses": 400, "T": 8, "s": 7.0, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 2400, "student_num_clauses": 400, "T": 8, "s": 7.0, "teacher_epochs": 30, "student_epochs": 30 },
    ]

    for i, params in enumerate(mnist_experiments):
        mnist_results, df = distillation_experiment(
            X_train, Y_train, X_test, Y_test, f"MNIST", params)
        print(mnist_results)

    """### Load Fashion MNIST data"""
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    # Data booleanization
    X_train = np.where(X_train > 75, 1, 0)
    X_test = np.where(X_test > 75, 1, 0)

    # Input data flattening
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()
    
    fashion_mnist_experiments = [
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 1600, "student_num_clauses": 400, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 2400, "student_num_clauses": 400, "T": 10, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
    ]
    
    for i, params in enumerate(mnist_experiments):
        fmnist_results, df = distillation_experiment(
            X_train, Y_train, X_test, Y_test, f"FashionMNIST", params)
        print(fmnist_results)


    """### Load KMNIST data"""
    train = KMNIST(root="data", download=True, train=True, transform=transforms.ToTensor())
    test = KMNIST(root="data", download=True, train=False, transform=transforms.ToTensor())

    X_train, Y_train = train.data.numpy(), train.targets.numpy()
    X_test, Y_test = test.data.numpy(), test.targets.numpy()

    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)

    X_train = np.where(X_train > 75, 1, 0)
    X_test = np.where(X_test > 75, 1, 0)

    kmnist_experiments = [
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 600, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 600, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 1600, "student_num_clauses": 400, "T": 600, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 2400, "student_num_clauses": 400, "T": 600, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
    ]
    
    for i, params in enumerate(kmnist_experiments):
        kmnist_results, df = distillation_experiment(
            X_train, Y_train, X_test, Y_test, f"KMNIST", params)
        print(kmnist_results)

    """### Load CIFAR-10 data
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        
    # Input data flattening
    X_train = X_train.reshape(X_train.shape[0], 32*32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32*32, 3)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    cifar10_experiments = [
        { "teacher_num_clauses": 400, "student_num_clauses": 100, "T": 500, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 800, "student_num_clauses": 100, "T": 500, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 1600, "student_num_clauses": 400, "T": 500, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
        { "teacher_num_clauses": 2400, "student_num_clauses": 400, "T": 500, "s": 5, "teacher_epochs": 30, "student_epochs": 30 },
    ]

    for i, params in enumerate(cifar10_experiments):
        cifar10_results, df = distillation_experiment(
            X_train, Y_train, X_test, Y_test, f"CIFAR-10", params)
        print(cifar10_results)
    """