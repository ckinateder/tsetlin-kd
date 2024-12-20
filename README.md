# tsetlin-kd

This is a demonstration of [knowledge distillation](https://arxiv.org/abs/1503.02531) using [Tsetlin Machines](https://arxiv.org/abs/1804.01508). This code is based on the [parallel Python implementation of a tsetlin machine](https://github.com/cair/pyTsetlinMachineParallel).

## Setup

### Build Docker Image

```bash
docker build -t tsetlin-kd .
```

### Run Docker Container

```bash
docker run -it --rm  -v $(pwd):/app tsetlin-kd bash
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

### Run the detailed example

```bash
python3 src/detailed_ex.py
```
