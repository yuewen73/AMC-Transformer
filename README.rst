AMC-Transformer
===============

This repository provides the reference implementation for the paper:

**AMC-Transformer: Automatic Modulation Classification based on Enhanced Attention Model**

The code is released for **research reproducibility purposes** and implements a
Transformer-based framework for automatic modulation classification (AMC) directly
from raw I/Q time-series signals.

Overview
--------

Automatic Modulation Classification (AMC) is a fundamental task in modern wireless
communication systems and is particularly relevant for spectrum awareness in future
6G networks.

This work explores the application of a multi-layer, multi-head self-attention
architecture to raw I/Q samples and provides a systematic empirical evaluation on
public AMC benchmarks.

Repository Structure
--------------------

::

    AMC-Transformer/
    ├── model.py              # AMC-Transformer (ViT-based) model definition
    ├── dataset.py            # RadioML 2018.01A dataset loading and preprocessing
    ├── train.py              # Training and evaluation entry point
    ├── requirements.txt      # Python dependencies
    └── README.rst

Dataset
-------

This implementation uses the **RadioML 2018.01A** dataset, which is publicly available.

- Dataset name: RadioML 2018.01A
- Official source: https://www.deepsig.ai/datasets

Please download the dataset separately and place it in the following directory
structure:

::

    data/
    └── RadioML2018/
        ├── GOLD_XYZ_OSC.0001_1024.hdf5
        └── classes-fixed.json

No dataset files are included in this repository.

Environment Setup
-----------------

The code is implemented in **Python** and uses **TensorFlow / Keras** for model
training and evaluation.  
Some auxiliary components (e.g., dataset handling) rely on PyTorch utilities.

Install the required dependencies using:

::

    pip install -r requirements.txt

The ``requirements.txt`` specifies the following minimum versions:

::

    numpy>=2.0.2
    pandas>=2.2.2
    tensorflow>=2.19.0
    h5py>=3.14.0
    scikit-learn>=1.6.1
    matplotlib>=3.10.0
    seaborn>=0.13.2
    torch>=2.8.0+cu126
    tqdm>=4.67.1

The experiments were tested with:

- Python >= 3.8
- TensorFlow >= 2.19
- NVIDIA GPU (optional, CPU execution is supported for reduced-scale runs)

Reproducibility
---------------

To reproduce a **representative experiment** reported in the paper
(e.g., SNR = 10 dB with a fixed random seed), run:

::

    python train.py \
        --data_dir ./data/RadioML2018 \
        --snr 10 \
        --seed 42 \
        --layers 10 \
        --heads 8

This command trains a Transformer model with 10 attention layers and 8 heads and
reports the classification accuracy on the test set.

For faster execution on a standard laptop or CPU-only environment, the following
optional arguments can be used:

::

    --max_samples 5000
    --epochs 20

These options limit the number of training samples and reduce training time while
preserving the overall performance trend discussed in the paper.

Due to computational constraints, this repository focuses on **representative runs**
rather than exhaustive hyperparameter sweeps or full multi-seed evaluations.

Notes
-----

- The provided code is intended for **academic and research use only**.
- All reproducible experiments should be executed via ``train.py``.
- Results may vary slightly depending on hardware, software versions, and random seeds.


License
-------

This project is released for **research and educational purposes**.
Please contact the authors for other usage scenarios.
