# AdaBoost with Support Vector Machine Classifier Written in Python
A supervised learning approach by applying algorithms and techniques of machine learning, to generate a predictive model for any applications with a focus on Support Vector Machines predictions.

Although initially the application was created for a predictive model for diagnosing breast cancer cells as either Benign or Malignant, the application was seen to be able to produce models with high accuracy for existing Support Vector Machine based classifications. Hence, as long as the dataset format consist of a combined training and testing set, with a single column containing labels $y_{i}  \in \left\{ -1, +1 \right\}$ and all features preceding each other, there is such use of applying the application onto the problem to produce a feasible predictive model.

**Live Working Demonstration:** [![Run on Repl.it](https://repl.it/badge/github/JarrenChov/AdaBoost-with-Support-Vector-Machine-Classifier)](https://repl.it/github/JarrenChov/AdaBoost-with-Support-Vector-Machine-Classifier)

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
      - [Applying Distribution Weight to SVM](#applying-distribution-weight-to-svm)
      - [Finding Significance Based Off Misclassification Error](#finding-significance-based-off-misclassification-error)
      - [Obtaining A New Distribution Weight](#obtaining-a-new-distribution-weight)
      - [Classifying Points Using The Prediction Model](#classifying-points-using-the-prediction-model)
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
- [Results](#results)
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
│   │   │   ├── check_application.py
│   │   │   └── check_type.py
│   │   ├── classes
│   │   │   ├── __init__.py
│   │   │   ├── classify.py
│   │   │   └── model.py
│   │   ├── get
│   │   │   ├── __init__.py
│   │   │   └── application_helper.py
│   │   │   └── dataset_default.py
│   │   │   └── extract_value.py
│   │   │   └── retrieve_param.py
│   │   ├── set
│   │   │   ├── __init__.py
│   │   │   └──set_param.py
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   ├── constants.py
│   │   ├── convert_type.py
│   │   └── format_dataset.py
│   ├── learning
│   │   ├── weak_learner
│   │   │   ├── classifier
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
│   ├── plotting
│   │   ├── __init__.py
│   │   ├── methods.py
│   │   └── plot_application.py
│   ├── test
│   │   ├── __init__.py
│   │   ├── test_check_type.py
│   │   ├── test_convert_type.py
│   │   ├── test_extract_value.py
│   │   ├── test_plotting_methods.py
│   │   ├── test_retrieve_param.py
│   │   └── test_set_param.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── application.py
│   ├── application_plot.py
│   └── methods.py
├── data
│   ├── unamed
│   │   ├── processed
│   │   │   ├── chunhua_shen_6000.csv
│   │   │   └── chunhua_shen_10000.csv
│   │   ├── raw
│   │   │   ├── test_data.csv
│   │   │   └── train_data.csv
│   │   ├── /tex
│   │   ├── README.md
│   │   └── README.tex.md
│   └── wdbc
│       ├── raw
│       │   └── wdbc_data.csv
│       ├── /tex
│       ├── README.md
│       └── README.tex.md
├── figs
│   ├── adaboost
│   │   └── step_details.png
│   └── plot
│       └── default_1_ss250_c10_ae8.png
├── /tex
├── .gitignore
├── .replit
├── Makefile
├── README.md
├── README.tex.md
└── requirements.txt
```
Please take note:

- All supplied data with corresponding information are located in `/data`
- All implementation details regarding AdaBoost are located in `/adaboost (main focus of this project)`
- All implementation details regarding PCA are located in `/adaboost/learning/dimension_reduction/pca`
- All implementation details regarding SVM are located in `/adaboost/learning/weak-learner/classifier/svm`
- All implementation details regarding plotting are located in `/adaboost/plotting`
- All class structure definitions are located in `/adaboost/common/classes`
- All unit tests are located in `/adaboost/test`
- All figures for analysis are located in `/fig`
- All commands are located in `Makefile`
- All required python dependencies and packages used are located in `requirements.txt`
- Any README.md is the now the output generated by [TeXify](https://github.com/agurodriguez/github-texify), after parsing LaTeX expressions into svg's from README.tex.md
- LaTeX expression svg's are stored in `/tex`

## Understanding the Concept of Adaptive Boosting
Adaptive Boosting is a way in which existing models can be furthered improved, by applying a series of weak learning algorithms with lower accuracy, with each weak learner learning from the previous learners mistakes, but combined together to obtain a single strong learner.

In addition, AdaBoost helps to solve the problem suffered from the “*curse of dimensionality*” where samples can span very large dimensions, reducing the ability to be able to construct a highly accurate and powerful model that can also easily be run in real time. Although, it has to be kept in mind that Adaptive Boosting is not bullet proof to such curse, where the dataset itself and the algorithm used as the weak-learner can constitutes such problem into over fitting. Such cause, can lead to over confidence in weak learners, which can effect the end accuracy as some learners are prioritized over others.

Nonetheless, the importance of each model relies in retrospect to its corresponding weight value. That is, the weight factors plays an important role and can vastly alter the distribution of the data and the next learning phase of the weak learner. Where incorrectly classified points populate a larger subset of the sample space due to a larger importance placed on correctly classifying such point.

By taking such importance into consideration, the weight values vastly fluctuate and the models change under the effects, such that:

- As an incorrectly classified sample keeps being incorrectly classified, the “weight” value will gradually increase, signifying the importance in correctly classifying such hard sample. Whereas the “weight” value will ever so increase towards 1.
- As an correctly classified sample keeps being correctly classified, the “weight” value will gradually decrease, signifying the unimportance in such sample. Where such sample will decrease towards 0 in the classifying stage, signifying a point which can be ignored.

### AdaBoost Algorithmic Implementation Details
Formulated from *Robert E. Schapire* original implementation [The boosting algorithm AdaBoost](#rob.schapire.net/papers/explaining-adaboost.pdf) on page 2, the AdaBoost algorithm is as described:

<img src="/figs/adaboost/step_details.png" width=850 height=400/>

Given such implementation details, to break it down into finer details step-by-step:
#### Applying Distribution Weight to SVM
Since each sample in the dataset is weighted to tis corresponding distribution weights $D_{t}$, such weight needs to be applied onto the Support Vector Machine implementation as described in [SVM Algorithmic Implementation Details](#svm-algorithmic-implementation-details). This is achieved, by slightly modifying the CVXOPT canonical form value `h` to include such weights.

Since h as described below, in the soft margin constitutes with a regulating parameter *$C$* for each sample, the matrix of regulating parameter constant *$C$* of dimensions *$datasetsamples \times 1$*, can be multiplied with the distribution weights $D_{t}$ corresponding to each sample. That is:
  > - *h is now an matrix of $0$’s corresponding to the dimensions of $a_{i}$ stacked horizontally below with another matrix of one multiplied by regulating parameter constant *$C$* multiplied by a distribution weight $D_{t}$.*

#### Finding Significance Based Off Misclassification Error
The misclassification error $\varepsilon_{t} = \frac{\sum_{i=1}^{m}w_{i}I\left ( h_{t}\left( x_{i} \neq y_{i} \right) \right )}{\sum_{i=1}^{m}w_{i}}$, at first glance may seem as a confusing mess of mathematical symbols, but hopefully this such explanation will clear things up. Initially $\sum_{i=1}^{m}w_{i}I\left ( h_{t}\left( x_{i} \neq y_{i} \right) \right)$ is just saying, *the misclassification error is the sum of all samples corresponding weight multiplied by the actual label multiplied by the prediction where the predicted label does not match the actual label*.

The part $h_{t}\left( x_{i} \neq y_{i} \right)$ *(prediction where the predicted label does not match the actual label)* might still be confusing, however if the mapping $h_{t} : X \rightarrow \left\{ -1, +1 \right\}$ is taken into account, all it means is:
- If a predicted label using $h_{t}\left( x_{i} \right)$ is **equal** to the actual label $y_{i}$, then the returned value is $false$ which is a $0$ in integer form.
- If a predicted label using $h_{t}\left( x_{i} \right)$ is **not equal** to the actual label $y_{i}$, then the returned value is $true$ which is a $1$ in integer form.

Using such information, it becomes apparent that all the formula is actually calculating as a percentage over all summed weights, the **summation of weights** where the prediction has **misclassified** a point.

Based off the misclassification error ($\varepsilon_{t}$), the significance ($\alpha_{t}$)  of the current weak learner (hypothesis) with the amount of final say in the strong classier, is determined by the equation $\alpha_{t} = \frac{1}{2}ln\left(  \frac{1 - \varepsilon_{t}}{\varepsilon_{t}} + \beta \right)$. The only aspect which differs from the original implementation by Robert E. Schapire is the introduction of a value $\beta$, due to the equation having a limitation when the misclassification error ($\varepsilon_{t}$) results in $0$ (remember...$log(0) = NaN$)! By ensuring such values can never reach 0, the value of $\beta$ as a threshold is introduced, such that $\beta = 1e-10$.

#### Obtaining A New Distribution Weight
Although this should be pretty straight forward, to keep the dataset as a uniform distribution of weights and ensuing all weights total $1$, the distribution weight is updated with  $D_{t} \leftarrow \frac{D_{t}\left ( i \right )\exp\left ( -\alpha_{i}y_{i}h_{t}\left ( x_{i} \right ) \right) }{Z_{t}}$. In addition, since Adaptive Boosting plays more importance on correctly classifying misclassified points, points that were correctly classified *(resulting in a negative exponential value)* are updated with a smaller weight of importance in the dataset. Whilst, on the other hand, points that were incorrectly classified *(resulting in a positive exponential value)* are updated with a larger weight of importance in the dataset.

Since the formula itself (the numerator) is not normalized as a uniform distribution, by using the equation $Z_{t} = \sum_{i=1}^{m}D_{t}\left ( i \right )\exp\left ( -\alpha_{i}y_{i}h_{t}\left ( x_{i} \right ) \right)$ each sample weight becomes a percentage of the total combined weights.

#### Classifying Points Using The Prediction Model
Since Adaptive Boosting uses $n$ number weak learners, also known as estimators, the ability to actually generate a singular model can be obtained by combining all weak learners into a singular model. The combination of all weak learners is expressed by $\sum_{t=1}^{t}\alpha_{t}h_{t}\left ( x \right )$, where the corresponding significance ($\alpha$) of the model is multiplied by the predicted label.

Since the point of interest lies in only knowing the sign of a corresponding label $ y_{i}  \in \left\{ -1, +1 \right\}$, the sign magnitude of the obtained prediction is taken such that $H\left ( x \right ) = sign\left ( \sum_{t=1}^{t}\alpha_{t}h_{t}\left ( x \right ) \right )$.

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
\begin{aligned}
\underset{a}{max}\sum_{i}^{m} a^{i} - \frac{1}
{2}\sum_{i,j}^{m}y^{i}y^{j}a^{i}a^{j}x^{i} \cdot x^{j}\\
\end{aligned}
$$

$$
\begin{aligned}
  subject \; to \quad & a_{i} \geq 0\\
  & \sum_{i}^{m} a^{i}y^{i} = 0\\
\end{aligned}
$$

To convert the problem into an a solvable CVXOPT canonical form of $\frac{1}{2}x^{T}Px \, + \, q^{T}x$, let $H_{i, j}$ to represent the matrix form of $y^{i}y^{j}x^{i} \cdot x^{j}$ , the dual form hence becomes of $\underset{a}{max}\sum_{i}^{m} a^{i} - \frac{1}{2}\sum_{i,j}^{m}a^{T}Ha$. In addition, to obtain the required form, the removal of summations through the use of vectors and inverses of the whole equation and conditions, turn a maximize problem into a minimize problem and required CVXOPT canonical form of:
$$
\begin{aligned}
\underset{a}{min} \: \frac{1}{2}a^{T}Ha-1^{T}a\\
\end{aligned}
$$

$$
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
\begin{aligned}
\underset{a}{min} \: \frac{1}{2}a^{T}Ha-1^{T}a
\end{aligned}
$$

$$
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
  & \text{+1} \quad \geq 0\\
  & \text{-1} \quad < 0\\
\end{cases}
\end{aligned}
$$

# Usage
For more detailed information when running the application use the following command:
```bash
python -m adaboost help
```
If you are using the make file to run the application, use the following command below:
```bash
make help
```
On the other hand, if a plot is wished to be generated of classification accuracy, it can be done s by running the following command:
```bash
python -m adaboost plot {parameters}
```
Note: Parameters is optional and follows the layout as described below in [Running the Application](#running-the-application). The most import part in running the application as a plot, is the inclusion of `plot` being specified in the arguments.

If you are however using the make file to run the application, use the following command below:
```bash
make run-plot
```
## Running the Application
To run the application, by initially starting at the root directory `AdaBoost-with-Support-Vector-Machine-Classifier`, run either of the the following commands to assign parameters in regards to AdaBoost, dataset file, PCA or SVM.

### User Defined Inputs
If you wish your enter in such parameters separately, use the command below to go through the process:
```bash
python -m adaboost
```
If you are using the make file to run the application, use the following command below:
>Note the below command only works if no parameters are specified and arguments are obtained via the application itself.

```bash
make run
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
    - default_# - Uses the supplied datasets present, where # corresponds to the datasets below:
      ( 1 ) Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

      ( 2 ) Unamed Dataset by Chunhua Shen [Subset sample - 6000 samples]

      ( 3 ) Unamed Dataset by Chunhua Shen [Full sample - 10000 samples]

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
  - Unspecified `<value>` will revert to default reduction string `<value=none>`
  - `<value>` can accept either of the following:
    - A string value *{default | none}* denoting either a default or no reduction to dataset
    - A float value in the range of *{0 <-> 1}* denoting a proportional reduction size to dataset
    - A integer value denoting a subset of a dataset
- `[*] svm_regularizer_c=<value>` *- specifies the boundary in misclassified points for a non-linearly separable dataset. A greater C values generates a more complex and tailored boundary to the data, whilst a lower C values generates a more generalized boundary*:
  - `<value>` can accept either of the following:
   - A string value *{default | none}* denoting either a default boundary, C=1.0 (soft-margin case) or cases in which all points lies on respective sides of a boundary (hard-margin case).
    - A float value denoting the boundary complexity and mis-classification of points.
- `[*] adaboost_estimators=<value>` *- specifies the number of AdaBoost generated prediction models (weak learners)*:
  - `<value>` can accept either of the following:
    - A string integer value denoting a number value
- `[#] output_detail=<value>` *- specifies verbose printing*:
  - Unspecified `<value>` will revert to default boolean `<value=false>`
  - `<value>` can accept either of the following:
    - A string boolean value *{true | false}* denoting a state
## Running the Application Test Suite
To run the tests to ensure the application is as bug free as possible, a series of tests can be run by initially starting at the root directory `AdaBoost-with-Support-Vector-Machine-Classifier`. To run the series of tests run the command below.
```bash
python -m unittest discover adaboost/test -v -b
```
If you are using the make file to run the application test suite, use the following command below:
```bash
make test
```
Although the supplied arguments are optional, the use of `-v` - verbose printing (Detail output) is to detail what current test is being run, and which part is exactly being tested, whilst *`-b` - buffer stdout and stderr* is used to suppress any application printouts causing clutter in the test suite itself. These itself, improves readability and clarity of tests and debugging if need be.

# Results
The results will be mainly focused on the Wisconsin Diagnostic Breast Cancer dataset, where the analysis will detail and explain the achieved prediction model accuracy for diagnosis breast cancer cells as either Benign or Malignant.

Since the Wisconsin Diagnostic Breast Cancer dataset consisted of 569 samples, the optimal separation of data into a training and testing set with a balance between accuracy and precision was 250 training samples (319 testing samples). Through testing, it was found that training such predictive model below 250 samples, would result with the accuracy of both the training and testing set would be substantially lower then a predictive model using only Support Vector Machines (SVM) as its basis. On the other hand, although training with 250 - 300 samples would produce accurate results with a fluctuation of $\pm 4$ between the highest and lowest accuracy of between both sets, any trained samples greater then the specified optimal range resulted in an over fitted model. where the accuracy of the training set would be significantly higher in accuracy with values in mid to high 90's, whilst the testing data would suffer in accuracy resulting in high 80's to low 90's.

Based on such model trained with 250 - 300 samples and varying parameter regulations $C$, as the regulating parameter $C$ was increased, with the SVM margin being more tailored to the training set, it was seen that there was a higher degree in accuracy with training reaching a maximum threshold of $96.8 \%$. Whilst, on the other hand, the testing set resulted in a significantly higher error rate then a normal Support Vector Machine model. On the contrary, with a lower regulating parameter $C$, both sets had a marginally higher error rate when compared to a SVM model.

To Determine and choose the optimal model to represent the Wisconsin Diagnostic Breast Cancer dataset, a model was chosen in which the following where met, with highest priority being listed first:
- A AdaBoost-SVM model where both sets achieve an accuracy greater then a model using only Support Vector Machine.
- A AdaBoost-SVM model where the fluctuations in errors between both sets are as minimal and as close as possible.
- A Model with a gradual decrement (Not instant) from initial Support Vector Machine model, to a error/accuracy theoretical maxima.

Hence, from these conclusions the predictive model with **Optimal Model Parameters** which was used to predict the diagnosis of breast cancer cells as either Benign or Malignant was as follows:
```bash
python -m adaboost dataset_file=default_1 dataset_sample_size=250 svm_regularizer_c=10 adaboost_estimators=8
```

This obtained model, achieved a 5% error rate on average between both sets, with both sets having the smallest deviation margin difference in error, whilst still obtaining a high classification rate in the mid $90 \%$, as shown below in Figure 1.

**Error Loss & Accuracy:**
<img src="/figs/plot/default_1_ss250_c10_ae8.png" width=750 height=450/>

*Figure 1: Graph showing error loss and model accuracy against number of iterators, using specified model parameters above.*

As a note, Since the dataset is rather small in terms of features, the amount of used weak-learners was also rather significantly smaller than initially thought. In addition, this may have also been effected as a Support Vector Machine classifier is rather an already *strong* classifier itself, with tendency to have higher then accuracy starting in the $80 \%$. Whilst, AdaBoost tends works better in generating predictive models with weaker classifiers that obtain $50 \% - 60 \%$ accuracy.
Although, generally an AdaBoost SVM model is trained using a RBF Kernel-SVM, this may be implemented in a future stage to see if the model can achieve a more stable and preforming model for future unseen cases.
