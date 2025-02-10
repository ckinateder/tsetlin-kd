# tsetlin-kd

This is a demonstration of [knowledge distillation](https://arxiv.org/abs/1503.02531) using [Tsetlin Machines](https://arxiv.org/abs/1804.01508). This code is based on the [parallel Python implementation of a tsetlin machine](https://github.com/cair/pyTsetlinMachineParallel).

## Setup

### Build Docker Image

```bash
docker build -t tsetlin-kd .
```

### Run Docker Container

```bash
docker run -it --rm  -v $(pwd):/app --name tskd tsetlin-kd bash
```

Ignore the following errors. We are not using CUDA.

```
2024-12-17 14:08:00.427838: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-12-17 14:08:00.428227: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-12-17 14:08:00.430272: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-12-17 14:08:00.435634: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1734444480.444560      21 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1734444480.447230      21 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-12-17 14:08:00.456452: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
```

## Usage

### Run the main file

```bash
python3 src/main.py
```

### Understanding the experiments

Here is an example of the `main.py` file:

```python3
from tensorflow.keras.datasets import mnist, fashion_mnist
from torchvision.datasets import KMNIST
from datasets import prepare_imdb_data
import h5py
import numpy as np
from distillation_experiment import distillation_experiment, downsample_experiment
from torchvision import transforms
import os
from datasets import MNISTDataset, MNIST3DDataset, FashionMNISTDataset, KMNISTDataset

# set seeds
np.random.seed(0)

if __name__ == "__main__":
    # load datasets
    kmnist_dataset = KMNISTDataset()
    mnist3d_dataset = MNIST3DDataset()
    mnist_dataset = MNISTDataset()
    fashion_mnist_dataset = FashionMNISTDataset()
    
    # run distilled experiments
    # this goes (dataset, name, params, kwargs)
    distilled_experiments = [
        (kmnist_dataset, "KMNIST", { "teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 600, "s": 5, "teacher_epochs": 20, "student_epochs": 60, "downsample": 0.02 }, {"overwrite": False}),
        (mnist3d_dataset, "MNIST3D", { "teacher_num_clauses": 2000, "student_num_clauses": 300, "T": 60, "s": 3.0, "teacher_epochs": 10, "student_epochs": 30, "downsample": 0.02 }, {"overwrite": False}),
        (mnist_dataset, "MNIST", { "teacher_num_clauses": 2000, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 80, "downsample": 0.02 }, {"overwrite": False}),
        (fashion_mnist_dataset, "FashionMNIST", { "teacher_num_clauses": 2000, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 80, "downsample": 0.02 }, {"overwrite": False}),
    ]
    
    for dataset, name, params, kwargs in distilled_experiments:
        distillation_experiment(dataset, name, params, **kwargs)

    # run downsample experiments
    # this goes (dataset, name, params, downsamples)
    downsample_experiments = [
        (mnist3d_dataset, "MNIST3D-Downsample", {"teacher_num_clauses": 2000, "student_num_clauses": 300, "T": 60, "s": 3.0, "teacher_epochs": 15, "student_epochs": 45 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]),
        (mnist_dataset, "MNIST-Downsample", {"teacher_num_clauses": 1200, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 80 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]),
        (kmnist_dataset, "KMNIST-Downsample", {"teacher_num_clauses": 1600, "student_num_clauses": 200, "T": 600, "s": 5, "teacher_epochs": 20, "student_epochs": 60 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]),
        (fashion_mnist_dataset, "FashionMNIST-Downsample", {"teacher_num_clauses": 2000, "student_num_clauses": 100, "T": 40, "s": 7.5,"teacher_epochs": 20, "student_epochs": 80 }, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]),
    ]
    
    for dataset, name, params, downsamples in downsample_experiments:
        downsample_experiment(dataset, name, params, downsamples)
```

**ADD MORE HERE**