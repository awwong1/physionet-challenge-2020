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

## Quickstart

```bash
# Retrieve the required git submodules
git submodule update --init evaluation-2020/
git submodule update --init apex/

# Download and extract the dataset
wget https://swift-yeg.cloud.cybera.ca:8080/v1/AUTH_e3b719b87453492086f32f5a66c427cf/physionet_2020/PhysioNetChallenge2020_Training_CPSC.tar.gz
md5sum --check MD5SUMS
# PhysioNetChallenge2020_Training_CPSC.tar.gz: OK
tar -xvf PhysioNetChallenge2020_Training_CPSC.tar.gz

# Initialize a Python Virtual Environment (venv/pip)
virtualenv venv --python python3
source venv/bin/activate
pip install -r requirements.txt
pip install ./apex/

# Run the classifier
python3 driver.py Training_WFDB out

# Evaluate the classifier scores
python3 evaluation-2020/evaluate_12ECG_score.py Training_WFDB out

# Sanity tests
python3 -m unittest discover
```

### Sanity checking the Dockerfile

```bash
docker build -t sanity .
docker run sanity:latest python3 driver.py Training_WFDB out
docker run -it -v Training_WFDB:/physionet/input_directory -v output_directory:/physionet/output_directory sanity bash

# within Docker context
python3 driver.py input_directory output_directory
exit
```

## Results

| Experiment | AUROC | AUPRC | Accuracy | F-measure | Fbeta-measure | Gbeta-measure |
|------------|-------|-------|----------|-----------|---------------|---------------|
| Baseline   | 0.506 | 0.128 | 0.831    | 0.046     | 0.071         | 0.029         |
| SimpleCNN  | 0.450 | 0.197 | 0.840    | 0.314     | 0.359         | 0.177         |

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
