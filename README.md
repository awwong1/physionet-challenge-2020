# Physionet Challenge 2020

Source code for the [2020 Physionet Challenge](https://physionetchallenges.github.io/2020/).

## Objective

From 12-lead ECG recordings, identify a set of one or more classes as well as a probability score for each class.

## Data

Download and extract [PhysioNetChallenge2020_Training_CPSC.tar.gz](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz). (`md5 8180611b87209d3897b0735a56780204`)

```bash
# tqdm in notebook
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab clean
jupyter lab build

# to test
python3 -m unittest discover

# Basic example (debugging locally)
python3 main.py configs/debug_simple_cnn.json

# SimpleCNN example
python3 main.py configs/preprocess_250hz_4000len.json  # converts *.hea and *.mat into *.npz files
python3 main.py configs/simple_cnn.json
```
