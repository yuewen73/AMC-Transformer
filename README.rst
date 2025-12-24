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
    ├── model.py              # AMC-Transformer model definition
    ├── dataset.py            # RadioML dataset loading and preprocessing
    ├── train.py              # Training and evaluation entry point
    ├── requirements.txt      # Python dependencies
    ├── configs/
        └── reproduce.yaml    # Representative configuration used in the paper


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

The code is implemented in **Python** and uses **TensorFlow / Keras**

Install the required dependencies using:

::

    pip install -r requirements.txt

The experiments were tested with:

- Python >= 3.8
- PyTorch >= 1.10

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

This configuration corresponds to the representative results discussed in the paper
and demonstrates the performance–complexity trade-off of the proposed AMC-Transformer
model.

Due to computational constraints, the repository focuses on **representative runs**
rather than exhaustive hyperparameter sweeps.

Notes
-----

- The provided code is intended for **academic and research use only**.
- Notebooks are included for exploratory analysis and visualization, but all
  reproducible experiments should be executed via ``train.py``.
- Results may vary slightly depending on hardware, software versions, and random seeds.

Citation
--------

If you use this code in your research, please cite the corresponding paper:

::

    @article{AMCTransformer2025,
      title   = {AMC-Transformer: Automatic Modulation Classification based on Enhanced Attention Model},
      author  = {Xu, Yuewen and others},
      journal = {Infocommunications Journal},
      year    = {2025}
    }

License
-------

This project is released for **research and educational purposes**.
Please contact the authors for other usage scenarios.

