# Installation

## 1. Install NerfStudio+Generfstudio

```bash
pip install git+https://github.com/hturki/generfstudio
```

Alternatively, clone then install this repo:
```bash
git clone https://github.com/hturki/generfstudio.git
cd generfstudio
pip install -e .
```

## 2. Install additional dependencies

```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html
```