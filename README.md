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
    pip install --upgrade pip
    pip install wheel
    pip install -r requirements.txt
    ```
- Download and extract the training data.
    ```bash
    cd data
    ./quickstart.sh
    ```
- Run the unittests.
    ```bash
    python3 -m unittest discover -v -s ./tests -p test_*.py
    ```

# Example Classifier

TODO: Update source code with proper submission attempt.

# Example classifier code for Python for the PhysioNet/CinC Challenge 2020

## Contents

This code uses two main scripts to train the model and classify the data:

* `train_model.py` Train your model. Add your model code to the `train_12ECG_model` function. It also performs all file input and output. **Do not** edit this script or we will be unable to evaluate your submission.
* `driver.py` is the classifier which calls the output from your `train_model` script. It also performs all file input and output. **Do not** edit this script or we will be unable to evaluate your submission.

Check the code in these files for the input and output formats for the `train_model` and `driver` scripts.

To create and save your model, you should edit `train_12ECG_classifier.py` script. Note that you should not change the input arguments of the `train_12ECG_classifier` function or add output arguments. The needed models and parameters should be saved in a separated file. In the sample code, an additional script, `get_12ECG_features.py`, is used to extract hand-crafted features.

To run your classifier, you should edit the `run_12ECG_classifier.py` script, which takes a single recording as input and outputs the predicted classes and probabilities. Please, keep the formats of both outputs as they are shown in the example. You should not change the inputs and outputs of the `run_12ECG_classifier` function.

## Use

You can run this classifier code by installing the requirements and running

    python train_model.py training_data model
    python driver.py model test_data test_outputs

where `training_data` is a directory of training data files, `model` is a directory of files for the model, `test_data` is the directory of test data files, and `test_outputs` is a directory of classifier outputs.  The [PhysioNet/CinC 2020 webpage](https://physionetchallenges.github.io/2020/) provides a training database with data files and a description of the contents and structure of these files.

## Submission

The `driver.py`, `get_12ECG_score.py`, and `get_12ECG_features.py` scripts must be in the root path of your repository. If they are inside a folder, then the submission will be unsuccessful.

## Details

See the [PhysioNet/CinC 2020 webpage](https://physionetchallenges.github.io/2020/) for more details, including instructions for the other files in this repository.
