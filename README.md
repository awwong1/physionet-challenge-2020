# Python Classifier Submission for PhysioNet/CinC Challenge 2020

Source code for the [PhysioNet/CinC Challenge 2020](https://physionetchallenges.github.io/2020/). Extends the [Python Classifier Template](https://github.com/physionetchallenges/python-classifier-2020).

## Team Details

Team name: **CVC**

- Alexander William Wong <[alex.wong@ualberta.ca](mailto:alex.wong@ualberta.ca)>
    - University of Alberta
- Weijie Sun <[weijie2@ualberta.ca](mailto:weijie2@ualberta.ca)>
    - Canadian Vigour Centre
- Sunil Kalmady Vasu <[kalmady@ualberta.ca](mailto:kalmady@ualberta.ca)>
    - Canadian Vigour Centre
- Padma Kaul <[pkaul@ualberta.ca](mailto:pkaul@ualberta.ca)>
    - Canadian Vigour Centre
- Abram Hindle <[abram.hindle@ualberta.ca](mailto:abram.hindle@ualberta.ca)>
    - University of Alberta

## Getting Started

All of the raw training data is available from [this discussion post](https://groups.google.com/d/msg/physionet-challenges/0ldKZgDGi0Y/sDPltA-EBAAJ).
- Retrieve the evaluation code
    ```bash
    git submodule update --init --recursive
    ```
- Create a python3 virtual environment and install the required dependencies.
    ```bash
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```
- Download and extract the training data.
    ```bash
    cd data
    ./quickstart.sh
    ```

# Example Classifier

TODO: Update source code with proper submission attempt.

## Contents

This classifier uses three scripts:

* `run_12ECG_classifier.py` makes the classification of the clinical 12-Leads ECG. Add your classification code to the `run_12ECG_classifier` function. It calls `get_12ECG_features.py` and to reduce your code's run time, add any code to the `load_12ECG_model` function that you only need to run once, such as loading weights for your model.
* `get_12ECG_features.py` extract the features from the clinical time-series data. This script and function are optional, but we have included it as an example.
* `driver.py` calls `load_12ECG_model` once and `run_12ECG_classifier` many times. Both functions are in `run_12ECG_classifier.py` file. This script also performs all file input and output. Please **do not** edit this script or we may be unable to evaluate your submission.

## Use

You can run this classifier by installing the packages in the `requirements.txt` file and running

    python driver.py input_directory output_directory

where `input_directory` is a directory for input data files and `output_directory` is a directory for output classification files. The PhysioNet/CinC 2020 webpage provides a training database with data files and a description of the contents and structure of these files.

## Submission

The `driver.py`, `run_12ECG_classifier.py`, and `get_12ECG_features.py` scripts need to be in the base or root path of the Github repository. If they are inside a subfolder, then the submission will fail.

## Details
“The baseline classifiers are simple logistic regression models. They use statistical moments of heart rate that we computed from the WFDB signal file (the `.mat` file) and demographic data taken directly from the WFDB header file (the `.hea` file) as predictors. 

The code uses a Python Online and Offline ECG QRS Detector based on the Pan-Tomkins algorithm. It was created and used for experimental purposes in psychophysiology and psychology. You can find more information in module documentation: https://github.com/c-labpl/qrs_detector
