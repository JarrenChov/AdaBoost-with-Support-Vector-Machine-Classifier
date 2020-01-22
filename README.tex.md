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
    - [SVM Hard Margin Case](#svm-hard-margin-case)
    - [SVM Soft Margin Case](#svm-soft-margin-case)
    - [SVM Algorithmic Implementation Details](#svm-algorithmic-implementation-details)
      - [SVM Dual Problem](#svm-dual-problem)
        - [Calculating Weights and Bias in The Dual Problem](#calculating-weights-and-bias-in-the-dual-problem)
        - [Classifying Points in The Dual Problem](#classifying-points-in-the-dual-problem)

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
│   │   ├── weak_learner
│   │   |   ├── classifier
│   │   │   │   ├── svm
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── application.py
│   │   │   │   │   └── methods.py
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   ├── dimension_reduction
│   │   │   ├── pca
│   │   │   │   ├── __init__.py
│   │   │   │   ├── application.py
│   │   │   │   └── methods.py
│   │   │   └── __init__.py
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
├── tex
├── .gitignore
├── README.md
└── README.tex.md
```
Please take note
- README.md is the now the output generated by [TeXify](github.com/agurodriguez/github-texify), after parsing LaTeX expressions into svg's from README.tex.md
- LaTeX expression svg's are stored in `/tex`
- All implementation details regarding AdaBoost are located in `/adaboost (main focus of this project)`
- All implementation details regarding PCA are located in `/adaboost/learning/dimension_reduction/pca`
- All implementation details regarding PCA are located in `/adaboost/learning/weak-learner/classifier/svm`
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

By using this *zero-centered data* '$X$', the covariance matrix $C_{x}$ which describes the variance between two pairs of points is calculated by using the formula:
$C_{x} = \frac{1}{n-1} X^{T} \cdot X$ , *where $n$ is the number of features*
> As a side note, Kernel Principal Component Analysis (KPCA) on the other hand, does not actually compute the eigenvalue and eigenvectors of a covariance space $C_{x}$, but instead on a projected space $\Phi (x)$ of the data. Thus, KPCA requires a mapping correlating to a $N$-dimensional space, where $N$ represent the number of data points (samples). That is, each data point $x_{i}$ is mapped to all existing data points to create such mapping in a high dimensional space, by creating such space using a kernel function $K$.
>
> Such that:  $\Phi (x_{i})$ maps to $\Phi : \mathbb{R}^{d} \rightarrow \mathbb{R}^{N}$, where $K$ = $K = k(x,y) = \left\langle \Phi(x),\Phi(y) \right\rangle = \Phi(x) \cdot \Phi(y^{T})$
> Where, $K$ can take the form of applying any kernel function such as:
> - Linear: $K(X_{n}, X_{m}) = X_{n} \cdot X_{m}^{T}$
> - Polynomial: $K(X_{n}, X_{m}) = \left( X_{n} \cdot X_{m}^{T} \right)^{d}$
> - Radial Basis Function: $K(X_{n}, X_{m}) = \exp\left( -\frac{\left\| X_{n} - X_{m}^{T} \right\|^{2}}{2\sigma^{2}} \right)$ or $\exp \left( -\gamma\left\| X_{n} - X_{m}^{T} \right\|^{2} \right)$
>   - As a note, Since the computational cost of calculating the Euclidean distance $\left\| X_{n} - X_{m}^{T} \right\|^{2}$ is time demanding, by using the $\ell^{2}$ norm, the computational time can be significantly reduced. Where $\left\| X_{n} - X_{m}^{T} \right\|_{2}^{2} = K(X_{n}, X_{n}) + K(X_{m}^{T}, X_{m}^{T}) -2K(X_{n}, X_{m}^{T})$
>
> However, since the above steps in PCA (mainly the mean calculation) do not apply to Kernel Principal Component Analysis, the newly mapped feature space requires data to be *zero-centered*, in which, can be achieved by normalizing the feature space $\Phi (x)$. Thus, the normalized feature space $K' = K - (1_{N} \cdot K) - (K \cdot 1_{N}) +  (1_{N} \cdot K \cdot 1_{N})$ and $1_{N}$ is a $N \times N$ matrix with all fields being $\frac{1}{N}$.

Using the obtained covariance matrix $C_{x}$, the the eigen decomposition of eigenvalues and eigenvectors can be obtained. By sorting obtained eigenvalues in descending order, with eigenvalues with largest variance ordered first, selecting the top $k$ eigenvalues results in a new matrix $R$ of shape $n \times k$. Where $R$ is a extraction of features to represent the reduced feature dataset.

By applying a projection of $P = D \times R$ onto the initial dataset, the resulting matrix $P$ of a $d \times k$ matrix, where d represents the original sample set rows, represents the new feature space of points relating to the initial data points.
## Understanding the Concept of Support Vector Machine
The importance and idea behind a support vector machine (SVM) is predicting the classification label of points contained within a set of data, by creating such a separating hyperplane of up to $n - 1$ dimensions, where $n$ is the total amount of classification constraints (features). By creating hyperplanes that allows for finite separation of such that points, the end goal is to find such only a single hyperplane which separates the classes with the highest width, with the width having an equidistance between the nearest points of the hyperplane line and the hyperplane itself.

Nonetheless, it is important to not oversee the importance of such points of the hyperplane, as merely just points. In addition, with even a single change in value, the effects can drastically alter the direction and position of an existing hyperplane. These points which are the *"support vectors"* (hence the name), are vital to generating a hyperplane of a support vector machine, where these points act as the fundamental *"pivotal points"* of a hyperplanes boundary.

These boundary however, can be of two categories, where one category draws only a linear line between classes, known as a hard margin case. Whilst another, tires to create a distinction between classes which are of non-linear separations, known as the soft margin case.
### SVM Hard Margin Case
The boundary of a hard margin is explicit and constitutes the “hard margin” of a linear support vector machine, also known as *"Hard Margin SVM"*. The general idea is a dataset where “all” points must be linearly separable into its resulting class, that is one side and one side only has a single class of points, whilst another side also only has a single class of points. This case however, allows for no mixed class separation when fitting a support vector, where any point that lies on the incorrect side of a hyperplane may results in failing to classify points by prediction.
### SVM Soft Margin Case
On the other hand, the case of points lying on the incorrect side of a hyperplane can be solved by modifying the boundaries to reflect these changes amd allowing such points to be on either side, by applying a regulating parameter *$C$*. This such case constitutes the “soft margin” of a non-linear support vector machine, also known as *"Soft Margin SVM"*. Consequently, the effect of such parameter "C" can vastly effect the accuracy of classifications and lead to the generalization problem, with whether a more accurate separator is deemed to be a better suite then a generalized separator.

That is to say, whether having a lower regulating parameter *$C$* value with less outlier points correctly classified, proportional to a larger max-margin width is a better end classification model that works for unseen data. Or a highly tuned model tailored with a larger regulating parameter *$C$*, with a inversely proportional max-margin width and high accuracy to seen data is deemed more important.

### SVM Algorithmic Implementation Details
Since the problems will be implemented in [CVXOPT](https://cvxopt.org/), the form for the dual need to be converted into canonical form as given by [quadratic programming form](https://cvxopt.org/userguide/coneprog.html#quadratic-programming) in CVXOPT api, such that the following are met:
$$
\begin{aligned}
  minimize \quad &\frac{1}{2}x^{T}Px \, + \, q^{T}x\\
  subject \; to \quad &Gx \leq h\\
  &Ax = b\\
\end{aligned}
$$

#### SVM Dual Problem
> As a side note, since the dual problem form is derived from the primal problem, a short glimpse.. actually maybe a dive.. into the primal form is given below.
> The primal form of a hard margin in which is specified by:
> $$
\begin{aligned}
  min \quad &\frac{1}{2} \left\| w \right\|^{2}\\
  subject \; to \quad &y_{i}\left( w^{T}x_{i} - b \right) \geq 1 \quad i=i,n\\
\end{aligned}
$$
>
> Furthermore, since it is required to solve for two variables, $w, b$, such variables can be combined into a singular matrix, such that $a = [w:b]$. Also, since the primal form is already quite marginally similar to the CVXOPT canonical form, the inverse of the conditions can be taken to end up in the required CVXOPT form of:
> $$
\begin{aligned}
  min \quad &\frac{1}{2} \left\| w \right\|^{2}\\
  subject \; to \quad &-y_{i}\left( w^{T}x_{i} - b \right) \leq -1 \quad i=i,n\\
\end{aligned}
$$
>
> In addition, given that it is intended to solve a soft margin primal problem, the regulating parameter *$C$* can be added such with a slack variable *"$\xi$"*, such that the new equation and constraints for a soft margin primal problem becomes:
> $$
\begin{aligned}
  min \quad &\frac{1}{2} \left\| w \right\|^{2} + C\sum_{i=1}^{m}\xi_{i}\\
  subject \; to \quad &-y_{i}\left( w^{T}x_{i} - b \right) \leq -1 + \xi_{i} \quad i=i,n\\
& -\xi_{i} \leq 0\\
\end{aligned}
$$

Derived from the primal problem, along with the lagrange multipliers $\pounds (a)$, the dual problem in the hard margin form is:
$$
\underset{a}{max}\sum_{i}^{m} a^{i} - \frac{1}
{2}\sum_{i,j}^{m}y^{i}y^{j}a^{i}a^{j}x^{i} \cdot x^{j}\newline
\begin{aligned}
  subject \; to \quad & a_{i} \geq 0\\
  & \sum_{i}^{m} a^{i}y^{i} = 0\\
\end{aligned}
$$

To convert the problem into an a solveable CVXOPT canonical form of $\frac{1}{2}x^{T}Px \, + \, q^{T}x$, let $H_{i, j}$ to represent the matrix form of $y^{i}y^{j}x^{i} \cdot x^{j}$ , the dual form hence becomes of $\underset{a}{max}\sum_{i}^{m} a^{i} - \frac{1}{2}\sum_{i,j}^{m}a^{T}Ha$. In addition, to obtain the required form, the removal of summations through the use of vectors and inverses of the whole equation and conditions, turn a maximize problem into a minimize problem and required CVXOPT canonical form of:
$$
\underset{a}{min} \: \frac{1}{2}a^{T}Ha-1^{T}a\newline
  \begin{aligned}
  subject \; to \quad & a_{i} \geq 0\\
  & y^{T}a = 0\\
\end{aligned}
$$

From this, by directly mapping corresponding values to the canonical form, values obtained are:
- P is a matrix is dimensions corresponding to $\left(\left( y \times x  \right) \cdot \left( y \times x  \right)^{T} \right)$
- q is a matrix of -1 with same dimensions but vertically as a single column
- G is a identity matrix corresponding to the dimensions of $a_{i}$
- h is a matrix of 0’s corresponding to the dimensions of $a_{i}$
- A is a matrix of the labels position rotated horizontally
- b is a matrix containing a single $0$

Furthermore, using the above form of a hard margin, a soft margin primal problem can be derived where a regulating parameter *$C$* is added,  such that another parameter of constraint is added to the conditions on $a_{i}$, such that $0 \leq a_{i} \leq C$. Such that now the condition $0 \leq a_{i}$ becomes $-a_{i} \leq 0$. Hence the soft margin dual problem form becomes:
$$
\underset{a}{min} \: \frac{1}{2}a^{T}Ha-1^{T}a\newline
\begin{aligned}
  subject \; to \quad & -a_{i} \geq 0\\
  & a_{i} \leq C\\
  & y^{T}a = 0\\
\end{aligned}
$$

In addition, with such new constraint, the canonical form parameters are modified such that:
- G is now an negative identity matrix corresponding to the dimensions of $a_{i}$ stacked below with another identity $I$ matrix corresponding to the dimensions of $a_{i}$
- h is now an matrix of $0$’s corresponding to the dimensions of $a_{i}$ stacked horizontally below with another matrix of one multiplied by regulating parameter constant *$C$*.

##### Calculating Weights and Bias in The Dual Problem
By solving this quadratic programming problem, the obtained results only resemble such lagrange mulitpliers $\alpha$. But to solve for $w, b$ the formula is constructed based of the derivation from the lagrange function of:
$$
\begin{aligned}
  & \pounds (w,b,a) = \frac{1}{2}w \cdot w - \sum_{i=1}^{m}a_{i}\left[ y_{i}(w \cdot x + b) -1 \right]
\end{aligned}
$$

By deriving, the obtained derivation for $w$:
$$
\begin{aligned}
  & w \rightarrow \triangledown_{w} \, \pounds (w,b,a) = w - \sum_{i=1}^{m}a_{i}y_{i}x_{i} = 0
\end{aligned}
$$

and by rearranging such equation, the formula for obtaining the value of $w$ becomes:
$$
\begin{aligned}
  & w \rightarrow w = \sum_{i=1}^{m}a_{i}y_{i}x_{i}
\end{aligned}
$$

On the other hand, we can obtain the value for $b$ by using the idea that $a_{i}\left[ y_{i}(w \cdot x + b) - 1 \right] = 0$, where $a$ gets cancelled out due to points being close to 0, leaving the form $y_{i}(w \cdot x + b) - 1 = 0$, where by cancelling each side by $y_{i} \rightarrow y_{i}^{2} = 1$, remains with $w \cdot x + b - y_{i}= 0$. Rearranging such equation with $b$ as the subject leaves (where $S$ is the support vector):
$$b = (y_{i} - w \cdot w)_{\in S}$$

##### Classifying Points in The Dual Problem
Since a support vector machine is of a linear separable hyperplane, which is of the form $y = ax + b$, once an hyperplane plane is obtained consisting of $w,b$, the form of $w \cdot x + b = 0$, which is the formula used to calculate the prediction of a classified point. In addition, the class which the prediction falls under is the idea that if we take the sign of the outcome. Hence, then the prediction becomes:
$$
\begin{aligned}
  & w \cdot w + b = \begin{cases}
  & \text{+1} \quad & \geq 0\\
  & \text{-1} \quad & < 0\\
\end{cases}
\end{aligned}
$$

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
- `[*] svm_regularizer_c=<value>` *- specifies the boundary in misclassified points for a non-linearly separable dataset. A greater C values generates a more complex and tailored boundary to the data, whilst a lower C values generates a more generalized boundary*:
  - `<value>` can accept either of the following:
   - A string value *{default / none}* denoting either a default boundary, C=1.0 (soft-margin case) or cases in which all points lies on respective sides of a boundary (hard-margin case).
    - A float value denoting the boundary complexity and mis-classification of points.
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
