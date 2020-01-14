# AdaBoost with Support-Vector Machine Classifier

 The project consists mainly of:
 - Unittest (Inbuilt Python Unit Testing)
 - Adaptive Boosting (AdaBoost)
 - Support Vector Machine (SVM) primarily focused on the dual form
 - Dimensionality Reduction using Principal Component Analysis (PCA)

# Table of Contents
- [General](#general)
  - [Directory Structure](#directory-structure)
  - [Understanding the concept of Adaptive Boosting (AdaBoost)](#understanding-the-concept-of-adaptive-boosting-(adaboost))
  - [Understanding the concept of Principal Component Analysis (PCA)](#understanding-the-concept-of-principal-component-analysis-(pca))
  - [Understanding the concept of Support Vector Machine (SVM)](#understanding-the-concept-of-support-vector-machine-(svm))
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Running the Application Test Suite](#running-the-application-test-suite)
# General
> Since this project was built using python3 (Python 3.6.9), please ensure that the project is run with at-least python3 (Python 3.5 to be safe). Although running the application with python2 may work (in some cases), some packages are built for python3 (Python 3.5 and above) and such, may lead to unexpected execution. or the application may outright not execute.
## Directory Structure
Initial directory structure without any make-shift changes or modifications are as specified:
```bash
AdaBoost-with-Support-Vector-Machine-Classifier
├── adaboost
│   ├── common
│   │   ├── check
│   │   │   ├── __init__.py
│   │   │   └── check_type.py
│   │   ├── get
│   │   │   ├── __init__.py
│   │   │   └── extract_value.py
│   │   │   └── retrieve_param.py
│   │   ├── set
│   │   │   ├── __init__.py
│   │   │   └──set_param.py
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   └── convert_type.py
│   ├── test
│   │   ├── __init__.py
│   │   ├── test_convert_type.py
│   │   ├── test_extract_value.py
│   │   ├── test_retrieve_param.py
│   │   └── test_set_param.py
│   ├── learning
│   │   ├── dimension_reduction
│   │   |   ├── pca
│   │   |   |   ├── application.py
│   │   |   |   └── __init__.py
│   │   |   └── __init__.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── __main__.py
│   └── application.py
├── data
│   ├── raw
│   │   └── wdbc_data.csv
│   └── README.md
├── .gitignore
└── README.md
```
Please take note
- All data with corresponding information are located in `/data`
- All unit tests are located in `/adaboost/test`
## Understanding the concept of Adaptive Boosting (AdaBoost)
## Understanding the concept of Principal Component Analysis (PCA)
## Understanding the concept of Support Vector Machine (SVM)
# Usage
## Running the Application
To run the application, by initially starting at the root directory `AdaBoost-with-Support-Vector-Machine-Classifier`, run either of the the following commands to assign parameters in regards to AdaBoost, dataset file, PCA or SVM.

If you wish your enter in such parameters separately, use the command below to go through the process:
```bash
python -m adaboost
```
One the other hand, if you wish to assign parameters straight from the command line, parameters can be a continuous set of **parameters written as lowercase, separated by a space between**, as listed below. Use the following command:
```bash
python -m adaboost {parameters}
```
Please note that:
> `[*]` - Denotes a required parameter. `[#]` - Denotes a optional parameter, where a default value will be overrided if such parameter is supplied.

Parameters allowed are of the following:
- `[*] dataset_file=<value>`:
  - `<value>` can accept either of the following:
    - default - Uses the supplied dataset(s) present *[Currently only a single dataset is present]*.
    - Any string path of a filepath leading to a dataset file *[Dataset type is not checked]*.
- `[*] dataset_sample_size=<value>`:
  - `<value>` can accept either of the following:
    - A string integer value denoting a number value
- `[*] adaboost_estimators=<value>`:
  - `<value>` can accept either of the following:
    - A string integer value denoting a number value
- `[#] output_detail=<value>`:
  - `<value>` can accept either of the following:
    - A string boolean value *{true / false}* denoting a state
## Running the Application Test Suite
To run the tests to ensure the application is as bug free as possible, a series of tests can be run by initially starting at the root directory `AdaBoost-with-Support-Vector-Machine-Classifier`. To run the series of tests run the command below.
```bash
python3 -m unittest discover adaboost/test -v -b
```
Although the supplied arguments are optional, the use of *`-v` - verbose printing (Detail output) is to detail what current test is being run, and which part is exactly being tested, whilst *`-b` - buffer stdout and stderr* is used to suppress any application printouts causing clutter in the test suite itself. These itself, improves readability and clarity of tests and debugging if need be.
