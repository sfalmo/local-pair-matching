### Setup

Working in a virtual environment is recommended.
Set one up with `python -m venv .venv`, activate it with `source .venv/bin/activate` and install the required packages with `pip install -r requirements.txt`.
To use a GPU with Tensorflow/Keras, refer to the corresponding section in the installation guide at [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip).

### Instructions

Simulation data can be found in `data` and trained models are located in `models`.
Sample script for training the considered models from scratch are given in `learn_c1.py`, `learn_fexc.py` and `learn_Fexc.py`.

### Further information

The reference data has been generated with grand canonical Monte Carlo simulations using [MBD](https://gitlab.uni-bayreuth.de/bt306964/mbd).
