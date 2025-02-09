from tensorflow.keras.datasets import mnist, fashion_mnist
from torchvision.datasets import KMNIST
from datasets import prepare_imdb_data
import h5py
import numpy as np
from distilled_experiment import distilled_experiment, downsample_experiment
from torchvision import transforms
import os
from datasets import MNISTDataset, MNIST3DDataset, FashionMNISTDataset, KMNISTDataset

# set seeds
np.random.seed(0)

if __name__ == "__main__":
    # run downsample experiment
    kmnist_dataset = KMNISTDataset()
    mnist3d_dataset = MNIST3DDataset()
    mnist_dataset = MNISTDataset()

    # MNIST
    mnist_ds_experiment = { 
        "teacher_num_clauses": 1200, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 60 }
    downsample_experiment(mnist_dataset, "MNIST-Downsample", params=mnist_ds_experiment, 
                          downsamples=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25], overwrite=True)
    
    # best T and s for MNIST are T=30,s=7.5
    # { "teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 40, "over": 1, "under": 0 },
    mnist_experiments = [        
        { "teacher_num_clauses": 2000, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 40, "downsample": 0.02 },
    ]

    for i, params in enumerate(mnist_experiments):
        mnist_results, df = distilled_experiment(
            mnist_dataset, f"MNIST", params)
        print(mnist_results)
    
    # MNIST-3D
    mnist3d_ds_experiment = { 
        "teacher_num_clauses": 2000, "student_num_clauses": 400, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 30, "downsample": 0.02 }
    downsample_experiment(mnist3d_dataset, "MNIST3D-Downsample", params=mnist3d_ds_experiment, 
                          downsamples=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35], overwrite=True)

    # so far, best params are: num_clauses=750, threshold=50, specificity=3.0
    #         { "teacher_num_clauses": 1000, "student_num_clauses": 100, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 20 },
    mnist3d_experiments = [
        { "teacher_num_clauses": 2000, "student_num_clauses": 400, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 30, "downsample": 0.02 },
    ]

    for i, params in enumerate(mnist3d_experiments):
        mnist3d_results, df = distilled_experiment(
            mnist3d_dataset, f"MNIST-3D", params)
        print(mnist3d_results)

    # KMNIST
    kmnist_experiments = [
        { "teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 600, "s": 5, "teacher_epochs": 30, "student_epochs": 30, "downsample": 0.02 },
    ]
    
    for i, params in enumerate(kmnist_experiments):
        kmnist_results, df = distilled_experiment(
            kmnist_dataset, f"KMNIST", params)
        print(kmnist_results)

    kmnist_ds_experiment = { 
        "teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 600, "s": 5, "teacher_epochs": 30, "student_epochs": 30, "downsample": 0.02 }
    downsample_experiment(kmnist_dataset, "KMNIST-Downsample", params=kmnist_ds_experiment, 
                          downsamples=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25], overwrite=True)
    
    # Fashion-MNIST
    
    

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
        mnist_results, df = distilled_experiment(
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
        mnist3d_results, df = distilled_experiment(
            X_train, Y_train, X_test, Y_test, f"MNIST-3D", params)
        print(mnist3d_results)

    """### Load IMDB data"""
    (X_train, Y_train), (X_test, Y_test) = prepare_imdb_data()
    imdb_experiments = [
        {"teacher_num_clauses": 10000, "student_num_clauses": 2000, "T": 80*100, "s": 10.0, "teacher_epochs": 30, "student_epochs": 60},
    ]
    
    for i, params in enumerate(imdb_experiments):
        imdb_results, df = distilled_experiment(
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
        mnist3d_results, df = distilled_experiment(
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
        mnist_results, df = distilled_experiment(
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
        fmnist_results, df = distilled_experiment(
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
        kmnist_results, df = distilled_experiment(
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
        cifar10_results, df = distilled_experiment(
            X_train, Y_train, X_test, Y_test, f"CIFAR-10", params)
        print(cifar10_results)
    """