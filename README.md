<<<<<<< HEAD
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
  - [Understanding the Concept of Adaptive Boosting](#understanding-the-concept-of-adaptive-boosting)
    - [AdaBoost Algorithmic Implementation Details](#adaboost-algorithmic-implementation-details)
  - [Understanding the Concept of Principal Component Analysis](#understanding-the-concept-of-principal-component-analysis)
    - [PCA Algorithmic Implementation Details](#pca-algorithmic-implementation-details)
  - [Understanding the Concept of Support Vector Machine](#understanding-the-concept-of-support-vector-machine)
    - [SVM Algorithmic Implementation Details](#svm-algorithmic-implementation-details)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
    - [User Defined Inputs](#user-defined-inputs)
    - [Argument Defined Inputs](#argument-defined-inputs)
  - [Running the Application Test Suite](#running-the-application-test-suite)
# Disclaimer
As a note, this project was purely made out of self-interest and dwelling into understanding the mathematical and algorithmic implementation details behind such algorithms and techniques. Where such methods have been applied and used to generate a model through supervised learning. Although using already existing inbuilt machine learning tools of python ([scikit-learn](https://scikit-learn.org/stable/) for those whom are unaware or are new) would be the best way to go and approach these types of projects in a real world scenario, this project was personally a way for myself to develop and harness techniques into creating efficient and simplistic programs in python for the end user, and if such arise, apply the these newly learnt techniques into further developed projects.

Although by following the implementation details (pseudo-code if applicable) and finding resources online into aiding the understanding of how to apply Adaptive Boosting (AdaBoost), Principal Component Analysis (PCA) and Support Vector Machines (SVM) in python without relying on already existing tools, some finer implementation details may be slightly incorrect or can have a variation. However, I have tried to keep the implementation as accurate as possible to its *'true'* implementation details . In such case, any pointers into creating a *'correct'* method of such would be highly appreciated, and as such further improve my knowledge aspect of machine learning, and better understand how to effectively apply machine learning techniques into producing an accurate as possible predictive model.
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
│   │   ├── test_check_type.py
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
## Understanding the Concept of Adaptive Boosting
### AdaBoost Algorithmic Implementation Details
placeholder text
## Understanding the Concept of Principal Component Analysis
The importance and idea behind Principal Component Analysis is to be able to extract such features from a vast range of features that represent/describe information of a given data. In which, the key features that highly describe such data extracted using Principal Component Analysis (PCA), allows for the conversion of such existing features from a possible high dimensional space into a lower linear space of principal components.

In addition, the key point to consider is that Principal Component Analysis is a measure of variance between features, where the importance lies in creating such availability, such that the variance can factor in the largest possible variety between data. Consequently, the degree of variance, and the its importance role in Principal Component Analysis is obtained through the use of Eigenvalues and Eigenvectors, where a vector is a direct representation of a direction, and its value underlies its variance of such data on its direction.

Another idea to keep in mind, is the fact that Principal Component Analysis works best for cases in which clear distinctions of classes can be generated on a linear plane, cases in which such distinctions of classes are cluttered together or even overlap in the same spot on a linear plane, will significantly impact the accuracy of the final produced model. To overcome this, PCA can be extended by applying the kernel method and applying popular kernel functions such as a Radial Basis Function or polynomial by mapping the classes into a non-linear space (This will be briefly touched upon in the implementation details section, with possibilities of implementation in a future release).
### PCA Algorithmic Implementation Details
Since Principal Component Analysis maps the features onto a linear plane, each initial sample data $D_i$ needs to be centered by subtracting a column-wise mean for each feature $n_x$ in the dataset, resulting in a *zero-centered data* '$X$'.

By using this *zero-centered data* '$X$', the covariance matrix $C_x$ which describes the variance between two pairs of points is calculated by using the formula:
$C_x$ = $\frac{1}{n - 1}$ $X^T$$\cdot$ $X$ , *where $n$ is the number of features*
> As a side note, Kernel Principal Component Analysis (KPCA) on the other hand, does not actually compute the eigenvalue and eigenvectors of a covariance space $C_x$, but instead on a projected space $\varPhi$$(x)$ of the data. Thus, KPCA requires a mapping correlating to a $N$-dimensional space, where $N$ represent the number of data points (samples). That is, each data point $x_i$ is mapped to all existing data points to create such mapping in a high dimensional space, by creating such space using a kernel function $K$.
>
> Such that: $\varPhi$$(X_i)$ maps to $\varPhi$ : $\R^d$ $\to$ $\R^N$, where $K$ = $k(x, y)$ = $\langle$$\varPhi(x)$, $\varPhi(y)$$\rangle$ = $\varPhi(x)$ $\cdot$ $\varPhi(y^T)$
> Where, $K$ can take the form of applying any kernel function such as:
> - Linear: $K(X_n, X_m)$ = $X_n$ $\cdot$ $X_m^T$
> - Polynomial: $K(X_n, X_m)$ = ($X_n$ $\cdot$ $X_m^T$$)^d$
> - Radial Basis Function: $K(X_n, X_m)$ = $\exp$$(-\frac{\|X_n - X_m^T\|^2}{2\sigma^2})$
>   - As a note, Since the computational cost of calculating the Euclidean distance $\|X_n - X_m^T\|^2$ is time demanding, by using the $\ell^2 norm$, the computational time can be significantly reduced. Where $\|X_n - X_m^T\|_2^2$ =  $K(X_n, X_n) + K(X_m^T, X_m^T) - 2K(X_n, X_m^T)$
>
> However, since the above steps in PCA (mainly the mean calculation) do not apply to Kernel Principal Component Analysis, the newly mapped feature space requires data to be *zero-centered*, in which, can be achieved by normalizing the feature space $\varPhi$$(x)$. Thus, the normalized feature space $K' = K - (1_N \cdot K) - (K \cdot 1_N) + (1_N \cdot K \cdot 1_N)$ and $1_N$ is a $N \times N$ matrix with all fields being $\frac{1}{N}$.

Using the obtained covariance matrix $C_x$, the the eigen decomposition of eigenvalues and eigenvectors can be obtained. By sorting obtained eigenvalues in descending order, with eigenvalues with largest variance ordered first, selecting the top $k$ eigenvalues results in a new matrix $R$ of shape $n \times k$. Where $R$ is a extraction of features to represent the reduced feature dataset.

By applying a projection of $P = D \times R$ onto the initial dataset, the resulting matrix $P$ of a $d \times k$ matrix, where d represents the original sample set rows, represents the new feature space of points relating to the initial data points.
## Understanding the Concept of Support Vector Machine
### SVM Algorithmic Implementation Details
placeholder text
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
> `[*]` - Denotes a required parameter, `[!*]` - denotes a required parameter given that the default dataset is not used, `[#]` - Denotes a optional parameter, where a default value will be overrided if such parameter is supplied.

Parameters in which will be processed are of the following:
- `[*] dataset_file=<value>` *- specifies a dataset file to be used or use default supplied file*:
  - `<value>` can accept either of the following:
    - default - Uses the supplied dataset(s) present *[Currently only a single dataset is present]*.
    - Any string path of a file-path leading to a dataset file *Note: [Dataset type is not checked, if using relative path, the directory starts at the root `AdaBoost-with-Support-Vector-Machine-Classifier`, .csv format required]*.
- `[*] dataset_sample_size=<value>` *- specifies a size to split the dataset into a training and testing set*:
  - `<value>` can accept either of the following:
    - A string integer value denoting a number value
- `[!*] dataset_label_column=<value>` *- specifies the column number for the dataset labels*:
  - `<value>` can accept either of the following:
    - A integer value denoting a column index in a dataset
- `[!*] dataset_feature_columns=<value> - <value>` *- specifies the range of columns dataset features span*:
  - `<value>` can accept either of the following:
    - A integer value denoting a column index in a dataset
- `[#] pca_reduction=<value>` *- specifies the size to reduce an existing dataset features to*:
  - Unspecified `<value>` will revert to default reduction string `<value=default>`
  - `<value>` can accept either of the following:
    - A string value *{default / none}* denoting either a default or no reduction to dataset
    - A float value in the range of *{0 <-> 1}* denoting a proportional reduction size to dataset
    - A integer value denoting a subset of a dataset
- `[*] adaboost_estimators=<value>` *- specifies the number of AdaBoost generated prediction models (weak learners)*:
  - `<value>` can accept either of the following:
    - A string integer value denoting a number value
- `[#] output_detail=<value>` *- specifies verbose printing*:
  - Unspecified `<value>` will revert to default boolean `<value=false>`
  - `<value>` can accept either of the following:
    - A string boolean value *{true / false}* denoting a state
## Running the Application Test Suite
To run the tests to ensure the application is as bug free as possible, a series of tests can be run by initially starting at the root directory `AdaBoost-with-Support-Vector-Machine-Classifier`. To run the series of tests run the command below.
```bash
python -m unittest discover adaboost/test -v -b
```
Although the supplied arguments are optional, the use of *`-v` - verbose printing (Detail output) is to detail what current test is being run, and which part is exactly being tested, whilst *`-b` - buffer stdout and stderr* is used to suppress any application printouts causing clutter in the test suite itself. These itself, improves readability and clarity of tests and debugging if need be.
=======
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
  - [Understanding the Concept of Adaptive Boosting](#understanding-the-concept-of-adaptive-boosting)
    - [AdaBoost Algorithmic Implementation Details](#adaboost-algorithmic-implementation-details)
  - [Understanding the Concept of Principal Component Analysis](#understanding-the-concept-of-principal-component-analysis)
    - [PCA Algorithmic Implementation Details](#pca-algorithmic-implementation-details)
  - [Understanding the Concept of Support Vector Machine](#understanding-the-concept-of-support-vector-machine)
    - [SVM Algorithmic Implementation Details](#svm-algorithmic-implementation-details)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
    - [User Defined Inputs](#user-defined-inputs)
    - [Argument Defined Inputs](#argument-defined-inputs)
  - [Running the Application Test Suite](#running-the-application-test-suite)
# Disclaimer
As a note, this project was purely made out of self-interest and dwelling into understanding the mathematical and algorithmic implementation details behind such algorithms and techniques. Where such methods have been applied and used to generate a model through supervised learning. Although using already existing inbuilt machine learning tools of python ([scikit-learn](https://scikit-learn.org/stable/) for those whom are unaware or are new) would be the best way to go and approach these types of projects in a real world scenario, this project was personally a way for myself to develop and harness techniques into creating efficient and simplistic programs in python for the end user, and if such arise, apply the these newly learnt techniques into further developed projects.

Although by following the implementation details (pseudo-code if applicable) and finding resources online into aiding the understanding of how to apply Adaptive Boosting (AdaBoost), Principal Component Analysis (PCA) and Support Vector Machines (SVM) in python without relying on already existing tools, some finer implementation details may be slightly incorrect or can have a variation. However, I have tried to keep the implementation as accurate as possible to its *'true'* implementation details . In such case, any pointers into creating a *'correct'* method of such would be highly appreciated, and as such further improve my knowledge aspect of machine learning, and better understand how to effectively apply machine learning techniques into producing an accurate as possible predictive model.
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
## Understanding the Concept of Adaptive Boosting
### AdaBoost Algorithmic Implementation Details
placeholder text
## Understanding the Concept of Principal Component Analysis
The importance and idea behind Principal Component Analysis is to be able to extract such features from a vast range of features that represent/describe information of a given data. In which, the key features that highly describe such data extracted using Principal Component Analysis (PCA), allows for the conversion of such existing features from a possible high dimensional space into a lower linear space of principal components.

In addition, the key point to consider is that Principal Component Analysis is a measure of variance between features, where the importance lies in creating such availability, such that the variance can factor in the largest possible variety between data. Consequently, the degree of variance, and the its importance role in Principal Component Analysis is obtained through the use of Eigenvalues and Eigenvectors, where a vector is a direct representation of a direction, and its value underlies its variance of such data on its direction.

Another idea to keep in mind, is the fact that Principal Component Analysis works best for cases in which clear distinctions of classes can be generated on a linear plane, cases in which such distinctions of classes are cluttered together or even overlap in the same spot on a linear plane, will significantly impact the accuracy of the final produced model. To overcome this, PCA can be extended by applying the kernel method and applying popular kernel functions such as a Radial Basis Function or polynomial by mapping the classes into a non-linear space (This will be briefly touched upon in the implementation details section, with possibilities of implementation in a future release).
### PCA Algorithmic Implementation Details
Since Principal Component Analysis maps the features onto a linear plane, each initial sample data <img src="/tex/3bdf20a1d3bb8900a92e3b28088057f1.svg?invert_in_darkmode&sanitize=true" align=middle width=18.26049554999999pt height=22.465723500000017pt/> needs to be centered by subtracting a column-wise mean for each feature <img src="/tex/322d8f61a96f4dd07a0c599482268dfe.svg?invert_in_darkmode&sanitize=true" align=middle width=17.32124954999999pt height=14.15524440000002pt/> in the dataset, resulting in a *zero-centered data* '<img src="/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>'.

By using this *zero-centered data* '<img src="/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>', the covariance matrix <img src="/tex/3ae57e069f94b2c7342d103c617510ae.svg?invert_in_darkmode&sanitize=true" align=middle width=19.203221399999993pt height=22.465723500000017pt/> which describes the variance between two pairs of points is calculated by using the formula:
<img src="/tex/3ae57e069f94b2c7342d103c617510ae.svg?invert_in_darkmode&sanitize=true" align=middle width=19.203221399999993pt height=22.465723500000017pt/> = <img src="/tex/d1e569c66f117ad4fffe206b7096ad59.svg?invert_in_darkmode&sanitize=true" align=middle width=24.952590299999994pt height=27.77565449999998pt/> <img src="/tex/b1e400cd7556070a5f713dc2121a83e9.svg?invert_in_darkmode&sanitize=true" align=middle width=24.44238554999999pt height=27.6567522pt/><img src="/tex/211dca2f7e396e7b572b4982e8ab3d19.svg?invert_in_darkmode&sanitize=true" align=middle width=4.5662248499999905pt height=14.611911599999981pt/> <img src="/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/> , *where <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is the number of features*
> As a side note, Kernel Principal Component Analysis (KPCA) on the other hand, does not actually compute the eigenvalue and eigenvectors of a covariance space <img src="/tex/3ae57e069f94b2c7342d103c617510ae.svg?invert_in_darkmode&sanitize=true" align=middle width=19.203221399999993pt height=22.465723500000017pt/>, but instead on a projected space <img src="/tex/795b51f0a6647864af7b860c73958533.svg?invert_in_darkmode&sanitize=true" align=middle width=10.958908949999989pt height=22.465723500000017pt/><img src="/tex/06b4c569ebdbb9d7575915683e7ee945.svg?invert_in_darkmode&sanitize=true" align=middle width=22.180421999999993pt height=24.65753399999998pt/> of the data. Thus, KPCA requires a mapping correlating to a <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/>-dimensional space, where <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> represent the number of data points (samples). That is, each data point <img src="/tex/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/> is mapped to all existing data points to create such mapping in a high dimensional space, by creating such space using a kernel function <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/>.
>
> Such that: <img src="/tex/795b51f0a6647864af7b860c73958533.svg?invert_in_darkmode&sanitize=true" align=middle width=10.958908949999989pt height=22.465723500000017pt/><img src="/tex/f6b58f5e674d0d39ba2ded1782bd4b2e.svg?invert_in_darkmode&sanitize=true" align=middle width=31.87698194999999pt height=24.65753399999998pt/> maps to <img src="/tex/795b51f0a6647864af7b860c73958533.svg?invert_in_darkmode&sanitize=true" align=middle width=10.958908949999989pt height=22.465723500000017pt/> : <img src="/tex/7598a56fca4704a1d383585919e1700c.svg?invert_in_darkmode&sanitize=true" align=middle width=6.843077999999991pt height=27.91243950000002pt/> <img src="/tex/e49c6dac8af82421dba6bed976a80bd9.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> <img src="/tex/96251c1ee8210f90349b6a3657b6f950.svg?invert_in_darkmode&sanitize=true" align=middle width=11.64616199999999pt height=27.6567522pt/>, where <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> = <img src="/tex/c4cff7400eec075645599a24c616eff0.svg?invert_in_darkmode&sanitize=true" align=middle width=47.21087249999999pt height=24.65753399999998pt/> = <img src="/tex/8a3003255478a8b4b4e52f2f212fd348.svg?invert_in_darkmode&sanitize=true" align=middle width=6.39271709999999pt height=24.65753399999998pt/><img src="/tex/f99053e10459ce92b25e86f3be0ad5bc.svg?invert_in_darkmode&sanitize=true" align=middle width=33.13933094999999pt height=24.65753399999998pt/>, <img src="/tex/841180d9d6d5f20c009dab7a8a44a58d.svg?invert_in_darkmode&sanitize=true" align=middle width=32.393549099999994pt height=24.65753399999998pt/><img src="/tex/50a9ef76c8bf8eb91244ed909528dac5.svg?invert_in_darkmode&sanitize=true" align=middle width=6.39271709999999pt height=24.65753399999998pt/> = <img src="/tex/f99053e10459ce92b25e86f3be0ad5bc.svg?invert_in_darkmode&sanitize=true" align=middle width=33.13933094999999pt height=24.65753399999998pt/> <img src="/tex/211dca2f7e396e7b572b4982e8ab3d19.svg?invert_in_darkmode&sanitize=true" align=middle width=4.5662248499999905pt height=14.611911599999981pt/> <img src="/tex/e5863f60535867b28a0f6cbce565faa1.svg?invert_in_darkmode&sanitize=true" align=middle width=42.74914214999999pt height=27.6567522pt/>
> Where, <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> can take the form of applying any kernel function such as:
> - Linear: <img src="/tex/4c5f13acf6a4b56b682be2c96e02fedf.svg?invert_in_darkmode&sanitize=true" align=middle width=83.90052164999999pt height=24.65753399999998pt/> = <img src="/tex/80f886ffbf0ed016ab2b1de28b34a791.svg?invert_in_darkmode&sanitize=true" align=middle width=21.74477414999999pt height=22.465723500000017pt/> <img src="/tex/211dca2f7e396e7b572b4982e8ab3d19.svg?invert_in_darkmode&sanitize=true" align=middle width=4.5662248499999905pt height=14.611911599999981pt/> <img src="/tex/ae4beaedfa92035ad7b9b770b370460c.svg?invert_in_darkmode&sanitize=true" align=middle width=25.28360174999999pt height=27.6567522pt/>
> - Polynomial: <img src="/tex/4c5f13acf6a4b56b682be2c96e02fedf.svg?invert_in_darkmode&sanitize=true" align=middle width=83.90052164999999pt height=24.65753399999998pt/> = (<img src="/tex/80f886ffbf0ed016ab2b1de28b34a791.svg?invert_in_darkmode&sanitize=true" align=middle width=21.74477414999999pt height=22.465723500000017pt/> <img src="/tex/211dca2f7e396e7b572b4982e8ab3d19.svg?invert_in_darkmode&sanitize=true" align=middle width=4.5662248499999905pt height=14.611911599999981pt/> <img src="/tex/ae4beaedfa92035ad7b9b770b370460c.svg?invert_in_darkmode&sanitize=true" align=middle width=25.28360174999999pt height=27.6567522pt/><img src="/tex/6d7f754453f5b1b00a21b70c482d43a8.svg?invert_in_darkmode&sanitize=true" align=middle width=13.23579509999999pt height=27.91243950000002pt/>
> - Radial Basis Function: <img src="/tex/4c5f13acf6a4b56b682be2c96e02fedf.svg?invert_in_darkmode&sanitize=true" align=middle width=83.90052164999999pt height=24.65753399999998pt/> = <img src="/tex/56f0fca36cc78d812ea01eda7c0d41a0.svg?invert_in_darkmode&sanitize=true" align=middle width=25.11424739999999pt height=14.15524440000002pt/><img src="/tex/4dc933c63387351fe6dc7831864e2f6e.svg?invert_in_darkmode&sanitize=true" align=middle width=100.21325985pt height=37.92139230000001pt/>
>   - As a note, Since the computational cost of calculating the Euclidean distance <img src="/tex/0b1d9c5990f9db4ef3ba15d77c68ec44.svg?invert_in_darkmode&sanitize=true" align=middle width=91.75436159999998pt height=27.6567522pt/> is time demanding, by using the <img src="/tex/fdf64e38041d184df0460af806c8a45c.svg?invert_in_darkmode&sanitize=true" align=middle width=54.36481049999999pt height=26.76175259999998pt/>, the computational time can be significantly reduced. Where <img src="/tex/5c01ddbcb374730f7f8d337cd3020927.svg?invert_in_darkmode&sanitize=true" align=middle width=91.75436159999998pt height=27.6567522pt/> =  <img src="/tex/2cd695b0f21f29effef736a14961fb81.svg?invert_in_darkmode&sanitize=true" align=middle width=300.10315664999996pt height=27.6567522pt/>
>
> However, since the above steps in PCA (mainly the mean calculation) do not apply to Kernel Principal Component Analysis, the newly mapped feature space requires data to be *zero-centered*, in which, can be achieved by normalizing the feature space <img src="/tex/795b51f0a6647864af7b860c73958533.svg?invert_in_darkmode&sanitize=true" align=middle width=10.958908949999989pt height=22.465723500000017pt/><img src="/tex/06b4c569ebdbb9d7575915683e7ee945.svg?invert_in_darkmode&sanitize=true" align=middle width=22.180421999999993pt height=24.65753399999998pt/>. Thus, the normalized feature space <img src="/tex/215f492ed649bf1fb50f3a228a6e581e.svg?invert_in_darkmode&sanitize=true" align=middle width=331.0813539pt height=24.7161288pt/> and <img src="/tex/12201befad62bfa952f550007a2e5d44.svg?invert_in_darkmode&sanitize=true" align=middle width=19.86537134999999pt height=21.18721440000001pt/> is a <img src="/tex/a964749a6b635295960fe89162eda4de.svg?invert_in_darkmode&sanitize=true" align=middle width=50.091150449999994pt height=22.465723500000017pt/> matrix with all fields being <img src="/tex/0bbb800d5e09a6f9df2ac4e715a64a9a.svg?invert_in_darkmode&sanitize=true" align=middle width=11.646161999999997pt height=27.77565449999998pt/>.

Using the obtained covariance matrix <img src="/tex/3ae57e069f94b2c7342d103c617510ae.svg?invert_in_darkmode&sanitize=true" align=middle width=19.203221399999993pt height=22.465723500000017pt/>, the the eigen decomposition of eigenvalues and eigenvectors can be obtained. By sorting obtained eigenvalues in descending order, with eigenvalues with largest variance ordered first, selecting the top <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> eigenvalues results in a new matrix <img src="/tex/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.60847334999999pt height=22.465723500000017pt/> of shape <img src="/tex/c0e991f2266d76861db938440931060c.svg?invert_in_darkmode&sanitize=true" align=middle width=39.03343619999999pt height=22.831056599999986pt/>. Where <img src="/tex/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.60847334999999pt height=22.465723500000017pt/> is a extraction of features to represent the reduced feature dataset.

By applying a projection of <img src="/tex/b18dcb4614c79272c92bad4c8cbade2c.svg?invert_in_darkmode&sanitize=true" align=middle width=81.52029764999999pt height=22.465723500000017pt/> onto the initial dataset, the resulting matrix <img src="/tex/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode&sanitize=true" align=middle width=12.83677559999999pt height=22.465723500000017pt/> of a <img src="/tex/0aa7f58b7e561001f5301aa03507f552.svg?invert_in_darkmode&sanitize=true" align=middle width=37.72252274999999pt height=22.831056599999986pt/> matrix, where d represents the original sample set rows, represents the new feature space of points relating to the initial data points.
## Understanding the Concept of Support Vector Machine
### SVM Algorithmic Implementation Details
placeholder text
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
> `[*]` - Denotes a required parameter, `[!*]` - denotes a required parameter given that the default dataset is not used, `[#]` - Denotes a optional parameter, where a default value will be overrided if such parameter is supplied.

Parameters in which will be processed are of the following:
- `[*] dataset_file=<value>` *- specifies a dataset file to be used or use default supplied file*:
  - `<value>` can accept either of the following:
    - default - Uses the supplied dataset(s) present *[Currently only a single dataset is present]*.
    - Any string path of a file-path leading to a dataset file *Note: [Dataset type is not checked, if using relative path, the directory starts at the root `AdaBoost-with-Support-Vector-Machine-Classifier`, .csv format required]*.
- `[*] dataset_sample_size=<value>` *- specifies a size to split the dataset into a training and testing set*:
  - `<value>` can accept either of the following:
    - A string integer value denoting a number value
- `[!*] dataset_label_column=<value>` *- specifies the column number for the dataset labels*:
  - `<value>` can accept either of the following:
    - A integer value denoting a column index in a dataset
- `[!*] dataset_feature_columns=<value> - <value>` *- specifies the range of columns dataset features span*:
  - `<value>` can accept either of the following:
    - A integer value denoting a column index in a dataset
- `[#] pca_reduction=<value>` *- specifies the size to reduce an existing dataset features to*:
  - Unspecified `<value>` will revert to default reduction string `<value=default>`
  - `<value>` can accept either of the following:
    - A string value *{default / none}* denoting either a default or no reduction to dataset
    - A float value in the range of *{0 <-> 1}* denoting a proportional reduction size to dataset
    - A integer value denoting a subset of a dataset
- `[*] adaboost_estimators=<value>` *- specifies the number of AdaBoost generated prediction models (weak learners)*:
  - `<value>` can accept either of the following:
    - A string integer value denoting a number value
- `[#] output_detail=<value>` *- specifies verbose printing*:
  - Unspecified `<value>` will revert to default boolean `<value=false>`
  - `<value>` can accept either of the following:
    - A string boolean value *{true / false}* denoting a state
## Running the Application Test Suite
To run the tests to ensure the application is as bug free as possible, a series of tests can be run by initially starting at the root directory `AdaBoost-with-Support-Vector-Machine-Classifier`. To run the series of tests run the command below.
```bash
python -m unittest discover adaboost/test -v -b
```
Although the supplied arguments are optional, the use of *`-v` - verbose printing (Detail output) is to detail what current test is being run, and which part is exactly being tested, whilst *`-b` - buffer stdout and stderr* is used to suppress any application printouts causing clutter in the test suite itself. These itself, improves readability and clarity of tests and debugging if need be.
>>>>>>> 26ba5fe677512ac994fc009bb924c36c99cebaa1
