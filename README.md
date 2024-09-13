# Neural density functionals: Local learning and pair-correlation matching

This repository contains code, datasets and models corresponding to the following publication:

**Neural density functionals: Local learning and pair-correlation matching**  
*Florian Sammüller and Matthias Schmidt, [Phys. Rev. E **110**, L032601](https://doi.org/10.1103/PhysRevE.110.L032601) (2024); [arXiv:2406.03327](https://arxiv.org/abs/2406.03327).*

### Setup

Working in a virtual environment is recommended.
Set one up with `python -m venv .venv`, activate it with `source .venv/bin/activate` and install the required packages with `pip install -r requirements.txt`.
To use a GPU with Tensorflow/Keras, refer to the corresponding section in the installation guide at [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip).

### Instructions

Simulation data can be found in `data` and trained models are located in `models`.
Sample scripts for training the considered models from scratch are given in `learn_c1.py`, `learn_fexc.py` and `learn_Fexc.py`.

### Further information

The reference data has been generated with grand canonical Monte Carlo simulations using [MBD](https://gitlab.uni-bayreuth.de/bt306964/mbd).
