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

