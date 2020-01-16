# AdaBoost with Support Vector Machine Classifier Written in Python
A supervised learning approach by applying algorithms and techniques of machine learning, to generate a predictive model for diagnosing breast cancer cells as either Benign or Malignant.

 The project consists mainly of:
 - Unittest (Inbuilt Python Unit Testing)
 - Adaptive Boosting (AdaBoost)
 - Support Vector Machine (SVM) primarily focused on the dual form
 - Dimensionality Reduction using Principal Component Analysis (PCA)

# Table of Contents
- [Disclaimer](#disclaimer)
- [General](#general)
  - [Directory Structure](#directory-structure)
  - [Understanding the concept of Adaptive Boosting (AdaBoost)](#understanding-the-concept-of-adaptive-boosting-(adaboost))
  - [Understanding the concept of Principal Component Analysis (PCA)](#understanding-the-concept-of-principal-component-analysis-(pca))
  - [Understanding the concept of Support Vector Machine (SVM)](#understanding-the-concept-of-support-vector-machine-(svm))
- [Usage](#usage)
  - [Running the Application](#running-the-application)
    - [User Defined Inputs](#user-defined-inputs)
    - [Argument Defined Inputs](#argument-defined-inputs)
  - [Running the Application Test Suite](#running-the-application-test-suite)
# Disclaimer
As a note, this project was purely made out of self-interest and dwelling into understanding the mathematical and algorithmic implementation details behind such algorithms and techniques. Where such methods have been applied and used to generate a model through supervised learning. Although using already existing inbuilt machine learning tools of python ([scikit-learn](https://scikit-learn.org/stable/) for those whom are unaware or are new) would be the best way to go and approach these types of projects in a real world scenario, this project was personally a way for myself to develop and harness techniques into creating efficient and simplistic programs in python for the end user, and if such arise, apply the these newly learnt techniques into further developed projects.

Although by following the implementation details (psuedo-code if applicable) and finding resources online into aiding the understanding of how to apply Adaptive Boosting (AdaBoost), Principal Component Analysis (PCA) and Support Vector Machines (SVM) in python without relying on already existing tools, some finer implementation details may be slightly incorrect or can have a varitation. However, I have tried to keep the implementation as accurate as possible to its *'true'* implementation details . In such case, any pointers into creating a *'correct'* method of such would be highly appreciated, and as such further improve my knowledge aspect of machine learning in a modern era, and help newcomers (like myself) into better understanding how to effectively apply machine learning techniques into producing an accurate as possible predictive model.
# General
> Since this project was built using python3 (Python 3.6.9), please ensure that the project is run with at-least python3 (Python 3.5 to be safe). Although running the application with python2 may work in some cases, some packages which were used are built explicitly for python3 (Python 3.5 and above) and such, may lead to unexpected execution, or the application may outright not execute.
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
│   │   ├── convert_type.py
│   │   └── format_dataset.py
│   ├── learning
│   │   ├── dimension_reduction
│   │   |   ├── pca
│   │   |   |   ├── application.py
│   │   |   |   ├── methods.py
│   │   |   |   └── __init__.py
│   │   |   └── __init__.py
│   │   └── __init__.py
│   ├── test
│   │   ├── __init__.py
│   │   ├── test_convert_type.py
│   │   ├── test_extract_value.py
│   │   ├── test_retrieve_param.py
│   │   └── test_set_param.py
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
- All implementation details regarding AdaBoost are located in `/adaboost (main focus of this project)`
- All implementation details regarding PCA are located in `/adaboost/learning/dimension_reduction/pca`
- All data with corresponding information are located in `/data`
- All unit tests are located in `/adaboost/test`
## Understanding the concept of Adaptive Boosting (AdaBoost)
## Understanding the concept of Principal Component Analysis (PCA)
## Understanding the concept of Support Vector Machine (SVM)
# Usage
## Running the Application
To run the application, by initially starting at the root directory `AdaBoost-with-Support-Vector-Machine-Classifier`, run either of the the following commands to assign parameters in regards to AdaBoost, dataset file, PCA or SVM.
### User Defined Inputs
If you wish your enter in such parameters separately, use the command below to go through the process:
```bash
python -m adaboost
```
### Argument Defined Inputs
One the other hand, if you wish to assign parameters straight from the command line, parameters can be a continuous set of **parameters written as lowercase, separated by a space between**, as listed below. Use the following command:
```bash
python -m adaboost {parameters}
```
Please note that:
> `[*]` - Denotes a required parameter. `[#]` - Denotes a optional parameter, where a default value will be overrided if such parameter is supplied.

Parameters in which will be processed are of the following:
- `[*] dataset_file=<value>`:
  - `<value>` can accept either of the following:
    - default - Uses the supplied dataset(s) present *[Currently only a single dataset is present]*.
    - Any string path of a filepath leading to a dataset file *[Dataset type is not checked]*.
- `[*] dataset_sample_size=<value>`:
  - `<value>` can accept either of the following:
    - A string integer value denoting a number value
- `[#] pca_reduction=<value>`:
  - Unspecified `<value>` will revert to default reduction string `<value=default>`
  - `<value>` can accept either of the following:
    - A string value *{default / none}* denoting either a default or no reduction to dataset
    - A float value in the range of *{0 <-> 1}* denoting a proportional reduction size to dataset
    - A integer value denoting a subset of a dataset
- `[*] adaboost_estimators=<value>`:
  - `<value>` can accept either of the following:
    - A string integer value denoting a number value
- `[#] output_detail=<value>`:
  - Unspecified `<value>` will revert to default boolean `<value=false>`
  - `<value>` can accept either of the following:
    - A string boolean value *{true / false}* denoting a state
## Running the Application Test Suite
To run the tests to ensure the application is as bug free as possible, a series of tests can be run by initially starting at the root directory `AdaBoost-with-Support-Vector-Machine-Classifier`. To run the series of tests run the command below.
```bash
python -m unittest discover adaboost/test -v -b
```
Although the supplied arguments are optional, the use of *`-v` - verbose printing (Detail output) is to detail what current test is being run, and which part is exactly being tested, whilst *`-b` - buffer stdout and stderr* is used to suppress any application printouts causing clutter in the test suite itself. These itself, improves readability and clarity of tests and debugging if need be.
