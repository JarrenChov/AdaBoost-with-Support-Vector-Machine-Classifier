# AdaBoost with Support Vector Machine Classifier Written in Python
A supervised learning approach by applying algorithms and techniques of machine learning, to generate a predictive model for any applications with a focus on Support Vector Machines predictions.

Although initially the application was created for a predictive model for diagnosing breast cancer cells as either Benign or Malignant, the application was seen to be able to produce models with high accuracy for existing Support Vector Machine based classifications. Hence, as long as the dataset format consist of a combined training and testing set, with a single column containing labels <img src="/tex/807acec709f43a1851a968ee98d33c65.svg?invert_in_darkmode&sanitize=true" align=middle width=99.37694744999997pt height=24.65753399999998pt/> and all features preceding each other, there is such use of applying the application onto the problem to produce a feasible predictive model.

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
Since each sample in the dataset is weighted to tis corresponding distribution weights <img src="/tex/bfc59112689000839ba3702dc1445221.svg?invert_in_darkmode&sanitize=true" align=middle width=18.575388149999988pt height=22.465723500000017pt/>, such weight needs to be applied onto the Support Vector Machine implementation as described in [SVM Algorithmic Implementation Details](#svm-algorithmic-implementation-details). This is achieved, by slightly modifying the CVXOPT canonical form value `h` to include such weights.

Since h as described below, in the soft margin constitutes with a regulating parameter *<img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>* for each sample, the matrix of regulating parameter constant *<img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>* of dimensions *<img src="/tex/53515155a3acfd1d452d1dd5980165a0.svg?invert_in_darkmode&sanitize=true" align=middle width=141.162747pt height=22.831056599999986pt/>*, can be multiplied with the distribution weights <img src="/tex/bfc59112689000839ba3702dc1445221.svg?invert_in_darkmode&sanitize=true" align=middle width=18.575388149999988pt height=22.465723500000017pt/> corresponding to each sample. That is:
  > - *h is now an matrix of <img src="/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>’s corresponding to the dimensions of <img src="/tex/0aae089ed20772138e327117bd8c6bac.svg?invert_in_darkmode&sanitize=true" align=middle width=13.340053649999989pt height=14.15524440000002pt/> stacked horizontally below with another matrix of one multiplied by regulating parameter constant *<img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>* multiplied by a distribution weight <img src="/tex/bfc59112689000839ba3702dc1445221.svg?invert_in_darkmode&sanitize=true" align=middle width=18.575388149999988pt height=22.465723500000017pt/>.*

#### Finding Significance Based Off Misclassification Error
The misclassification error <img src="/tex/1be0b205f74b46752f6c3239db418f5c.svg?invert_in_darkmode&sanitize=true" align=middle width=162.41362005pt height=34.8495345pt/>, at first glance may seem as a confusing mess of mathematical symbols, but hopefully this such explanation will clear things up. Initially <img src="/tex/7496894d0d4f1d3984b752d0cb66e3a8.svg?invert_in_darkmode&sanitize=true" align=middle width=164.5918758pt height=26.438629799999987pt/> is just saying, *the misclassification error is the sum of all samples corresponding weight multiplied by the actual label multiplied by the prediction where the predicted label does not match the actual label*.

The part <img src="/tex/8153cc72912947bba38234b7c3295c56.svg?invert_in_darkmode&sanitize=true" align=middle width=81.10153589999999pt height=24.65753399999998pt/> *(prediction where the predicted label does not match the actual label)* might still be confusing, however if the mapping <img src="/tex/d9a58b487fa06753e6d0f924870953ba.svg?invert_in_darkmode&sanitize=true" align=middle width=135.19007204999997pt height=24.65753399999998pt/> is taken into account, all it means is:
- If a predicted label using <img src="/tex/8e12aea9d06260148994f9684ecc3a76.svg?invert_in_darkmode&sanitize=true" align=middle width=45.651676949999995pt height=24.65753399999998pt/> is **equal** to the actual label <img src="/tex/e46f5aa3f3f039ebf21a80fd0cf8fad9.svg?invert_in_darkmode&sanitize=true" align=middle width=12.710331149999991pt height=14.15524440000002pt/>, then the returned value is <img src="/tex/880e3fa8f992a726d9df913ed6910955.svg?invert_in_darkmode&sanitize=true" align=middle width=39.09452909999999pt height=22.831056599999986pt/> which is a <img src="/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> in integer form.
- If a predicted label using <img src="/tex/8e12aea9d06260148994f9684ecc3a76.svg?invert_in_darkmode&sanitize=true" align=middle width=45.651676949999995pt height=24.65753399999998pt/> is **not equal** to the actual label <img src="/tex/e46f5aa3f3f039ebf21a80fd0cf8fad9.svg?invert_in_darkmode&sanitize=true" align=middle width=12.710331149999991pt height=14.15524440000002pt/>, then the returned value is <img src="/tex/39f91fc3a8d40698207a280c10017638.svg?invert_in_darkmode&sanitize=true" align=middle width=30.87346349999999pt height=20.221802699999984pt/> which is a <img src="/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> in integer form.

Using such information, it becomes apparent that all the formula is actually calculating as a percentage over all summed weights, the **summation of weights** where the prediction has **misclassified** a point.

Based off the misclassification error (<img src="/tex/5e38c7d440020e7fc6642fe0b2dd13dd.svg?invert_in_darkmode&sanitize=true" align=middle width=12.631296149999992pt height=14.15524440000002pt/>), the significance (<img src="/tex/9e2b8d7af10391275b0cbe00525e139f.svg?invert_in_darkmode&sanitize=true" align=middle width=15.48143849999999pt height=14.15524440000002pt/>)  of the current weak learner (hypothesis) with the amount of final say in the strong classier, is determined by the equation <img src="/tex/f72282bcb6464c41848571505b2fd198.svg?invert_in_darkmode&sanitize=true" align=middle width=148.88218784999998pt height=37.80850590000001pt/>. The only aspect which differs from the original implementation by Robert E. Schapire is the introduction of a value <img src="/tex/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode&sanitize=true" align=middle width=10.16555099999999pt height=22.831056599999986pt/>, due to the equation having a limitation when the misclassification error (<img src="/tex/5e38c7d440020e7fc6642fe0b2dd13dd.svg?invert_in_darkmode&sanitize=true" align=middle width=12.631296149999992pt height=14.15524440000002pt/>) results in <img src="/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> (remember...<img src="/tex/9914f48b211435bdd696e9aeb6e1065b.svg?invert_in_darkmode&sanitize=true" align=middle width=103.23814049999999pt height=24.65753399999998pt/>)! By ensuring such values can never reach 0, the value of <img src="/tex/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode&sanitize=true" align=middle width=10.16555099999999pt height=22.831056599999986pt/> as a threshold is introduced, such that <img src="/tex/2c3f82af26e6e76622c701d730c9fbc2.svg?invert_in_darkmode&sanitize=true" align=middle width=84.48611985pt height=22.831056599999986pt/>.

#### Obtaining A New Distribution Weight
Although this should be pretty straight forward, to keep the dataset as a uniform distribution of weights and ensuing all weights total <img src="/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>, the distribution weight is updated with  <img src="/tex/c434f730623033124f84fa37d41e2164.svg?invert_in_darkmode&sanitize=true" align=middle width=182.5870563pt height=33.20539859999999pt/>. In addition, since Adaptive Boosting plays more importance on correctly classifying misclassified points, points that were correctly classified *(resulting in a negative exponential value)* are updated with a smaller weight of importance in the dataset. Whilst, on the other hand, points that were incorrectly classified *(resulting in a positive exponential value)* are updated with a larger weight of importance in the dataset.

Since the formula itself (the numerator) is not normalized as a uniform distribution, by using the equation <img src="/tex/2a4740f8512c750fd9474ef8ee09e233.svg?invert_in_darkmode&sanitize=true" align=middle width=253.05713894999994pt height=26.438629799999987pt/> each sample weight becomes a percentage of the total combined weights.

#### Classifying Points Using The Prediction Model
Since Adaptive Boosting uses <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> number weak learners, also known as estimators, the ability to actually generate a singular model can be obtained by combining all weak learners into a singular model. The combination of all weak learners is expressed by <img src="/tex/ccd6e267dbf772a0805711c9da32a32d.svg?invert_in_darkmode&sanitize=true" align=middle width=99.00512324999998pt height=30.685221600000023pt/>, where the corresponding significance (<img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>) of the model is multiplied by the predicted label.

Since the point of interest lies in only knowing the sign of a corresponding label <img src="/tex/36f03ec0af72b674125fc46075f844fa.svg?invert_in_darkmode&sanitize=true" align=middle width=99.37694744999997pt height=24.65753399999998pt/>, the sign magnitude of the obtained prediction is taken such that <img src="/tex/50995c0d17887710ea6207bc63d8df39.svg?invert_in_darkmode&sanitize=true" align=middle width=214.88309699999994pt height=37.80850590000001pt/>.

## Understanding the Concept of Principal Component Analysis
The importance and idea behind Principal Component Analysis is to be able to extract such features from a vast range of features that represent/describe information of a given data. In which, the key features that highly describe such data extracted using Principal Component Analysis (PCA), allows for the conversion of such existing features from a possible high dimensional space into a lower linear space of principal components.

In addition, the key point to consider is that Principal Component Analysis is a measure of variance between features, where the importance lies in creating such availability, such that the variance can factor in the largest possible variety between data. Consequently, the degree of variance, and the its importance role in Principal Component Analysis is obtained through the use of Eigenvalues and Eigenvectors, where a vector is a direct representation of a direction, and its value underlies its variance of such data on its direction.

Another idea to keep in mind, is the fact that Principal Component Analysis works best for cases in which clear distinctions of classes can be generated on a linear plane, cases in which such distinctions of classes are cluttered together or even overlap in the same spot on a linear plane, will significantly impact the accuracy of the final produced model. To overcome this, PCA can be extended by applying the kernel method and applying popular kernel functions such as a Radial Basis Function or polynomial by mapping the classes into a non-linear space (This will be briefly touched upon in the implementation details section, with possibilities of implementation in a future release).
### PCA Algorithmic Implementation Details
Since Principal Component Analysis maps the features onto a linear plane, each initial sample data <img src="/tex/3bdf20a1d3bb8900a92e3b28088057f1.svg?invert_in_darkmode&sanitize=true" align=middle width=18.26049554999999pt height=22.465723500000017pt/> needs to be centered by subtracting a column-wise mean for each feature <img src="/tex/322d8f61a96f4dd07a0c599482268dfe.svg?invert_in_darkmode&sanitize=true" align=middle width=17.32124954999999pt height=14.15524440000002pt/> in the dataset, resulting in a *zero-centered data* '<img src="/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>'.

By using this *zero-centered data* '<img src="/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/>', the covariance matrix <img src="/tex/be5feb25c1beeb819cf19dbeb233e085.svg?invert_in_darkmode&sanitize=true" align=middle width=19.203221399999993pt height=22.465723500000017pt/> which describes the variance between two pairs of points is calculated by using the formula:
<img src="/tex/689abbe36122a6fd09d08fa4a662ccf1.svg?invert_in_darkmode&sanitize=true" align=middle width=122.88543134999998pt height=27.77565449999998pt/> , *where <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is the number of features*
> As a side note, Kernel Principal Component Analysis (KPCA) on the other hand, does not actually compute the eigenvalue and eigenvectors of a covariance space <img src="/tex/be5feb25c1beeb819cf19dbeb233e085.svg?invert_in_darkmode&sanitize=true" align=middle width=19.203221399999993pt height=22.465723500000017pt/>, but instead on a projected space <img src="/tex/5de88e3c940231fe84422b37f7eff3a3.svg?invert_in_darkmode&sanitize=true" align=middle width=34.05260099999999pt height=24.65753399999998pt/> of the data. Thus, KPCA requires a mapping correlating to a <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/>-dimensional space, where <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> represent the number of data points (samples). That is, each data point <img src="/tex/dc80c8df8d6a3120a158fb62653b1321.svg?invert_in_darkmode&sanitize=true" align=middle width=14.045887349999989pt height=14.15524440000002pt/> is mapped to all existing data points to create such mapping in a high dimensional space, by creating such space using a kernel function <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/>.
>
> Such that:  <img src="/tex/f191b6487df2053d06e17b914291f443.svg?invert_in_darkmode&sanitize=true" align=middle width=39.52539689999999pt height=24.65753399999998pt/> maps to <img src="/tex/31b87e9f72707101fc2a029d2a43e434.svg?invert_in_darkmode&sanitize=true" align=middle width=94.19671634999999pt height=27.91243950000002pt/>, where <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> = <img src="/tex/efaa46dbbaa4c62c3c9d9c618ccd51bf.svg?invert_in_darkmode&sanitize=true" align=middle width=305.1384039pt height=27.6567522pt/>
> Where, <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> can take the form of applying any kernel function such as:
> - Linear: <img src="/tex/eef379cc44df26d81ee2d70283e8ed12.svg?invert_in_darkmode&sanitize=true" align=middle width=165.54042449999997pt height=27.6567522pt/>
> - Polynomial: <img src="/tex/c04ff4dde718eb012980cef877e4e170.svg?invert_in_darkmode&sanitize=true" align=middle width=188.27395619999996pt height=35.79931739999998pt/>
> - Radial Basis Function: <img src="/tex/48ace73499133653fc19f12ac8899b2e.svg?invert_in_darkmode&sanitize=true" align=middle width=248.2689462pt height=47.6716218pt/> or <img src="/tex/a158f83a25339abced612a54e9e15943.svg?invert_in_darkmode&sanitize=true" align=middle width=166.8402879pt height=37.80850590000001pt/>
>   - As a note, Since the computational cost of calculating the Euclidean distance <img src="/tex/5a66ddf43c29a3ee88e61e33543197f7.svg?invert_in_darkmode&sanitize=true" align=middle width=93.58079444999998pt height=34.64868000000003pt/> is time demanding, by using the <img src="/tex/97132f2574611b6bad3cd4caa5e3ec21.svg?invert_in_darkmode&sanitize=true" align=middle width=13.40191379999999pt height=26.76175259999998pt/> norm, the computational time can be significantly reduced. Where <img src="/tex/a65e8451b923f77d14dbf7f3ca3db928.svg?invert_in_darkmode&sanitize=true" align=middle width=416.4234954pt height=34.64868000000003pt/>
>
> However, since the above steps in PCA (mainly the mean calculation) do not apply to Kernel Principal Component Analysis, the newly mapped feature space requires data to be *zero-centered*, in which, can be achieved by normalizing the feature space <img src="/tex/5de88e3c940231fe84422b37f7eff3a3.svg?invert_in_darkmode&sanitize=true" align=middle width=34.05260099999999pt height=24.65753399999998pt/>. Thus, the normalized feature space <img src="/tex/493a06adcbfa8324adffcb171c1352ea.svg?invert_in_darkmode&sanitize=true" align=middle width=331.0813539pt height=24.7161288pt/> and <img src="/tex/a3a93ba00c04c658e06a4ca86205b2c7.svg?invert_in_darkmode&sanitize=true" align=middle width=19.86537134999999pt height=21.18721440000001pt/> is a <img src="/tex/a964749a6b635295960fe89162eda4de.svg?invert_in_darkmode&sanitize=true" align=middle width=50.091150449999994pt height=22.465723500000017pt/> matrix with all fields being <img src="/tex/0bbb800d5e09a6f9df2ac4e715a64a9a.svg?invert_in_darkmode&sanitize=true" align=middle width=11.646161999999997pt height=27.77565449999998pt/>.

Using the obtained covariance matrix <img src="/tex/be5feb25c1beeb819cf19dbeb233e085.svg?invert_in_darkmode&sanitize=true" align=middle width=19.203221399999993pt height=22.465723500000017pt/>, the the eigen decomposition of eigenvalues and eigenvectors can be obtained. By sorting obtained eigenvalues in descending order, with eigenvalues with largest variance ordered first, selecting the top <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> eigenvalues results in a new matrix <img src="/tex/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.60847334999999pt height=22.465723500000017pt/> of shape <img src="/tex/c0e991f2266d76861db938440931060c.svg?invert_in_darkmode&sanitize=true" align=middle width=39.03343619999999pt height=22.831056599999986pt/>. Where <img src="/tex/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.60847334999999pt height=22.465723500000017pt/> is a extraction of features to represent the reduced feature dataset.

By applying a projection of <img src="/tex/b18dcb4614c79272c92bad4c8cbade2c.svg?invert_in_darkmode&sanitize=true" align=middle width=81.52029764999999pt height=22.465723500000017pt/> onto the initial dataset, the resulting matrix <img src="/tex/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode&sanitize=true" align=middle width=12.83677559999999pt height=22.465723500000017pt/> of a <img src="/tex/0aa7f58b7e561001f5301aa03507f552.svg?invert_in_darkmode&sanitize=true" align=middle width=37.72252274999999pt height=22.831056599999986pt/> matrix, where d represents the original sample set rows, represents the new feature space of points relating to the initial data points.

## Understanding the Concept of Support Vector Machine
The importance and idea behind a support vector machine (SVM) is predicting the classification label of points contained within a set of data, by creating such a separating hyperplane of up to <img src="/tex/3eeee545b1fbecf1f5a508b7304d7d5c.svg?invert_in_darkmode&sanitize=true" align=middle width=38.17727759999999pt height=21.18721440000001pt/> dimensions, where <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is the total amount of classification constraints (features). By creating hyperplanes that allows for finite separation of such that points, the end goal is to find such only a single hyperplane which separates the classes with the highest width, with the width having an equidistance between the nearest points of the hyperplane line and the hyperplane itself.

Nonetheless, it is important to not oversee the importance of such points of the hyperplane, as merely just points. In addition, with even a single change in value, the effects can drastically alter the direction and position of an existing hyperplane. These points which are the *"support vectors"* (hence the name), are vital to generating a hyperplane of a support vector machine, where these points act as the fundamental *"pivotal points"* of a hyperplanes boundary.

These boundary however, can be of two categories, where one category draws only a linear line between classes, known as a hard margin case. Whilst another, tires to create a distinction between classes which are of non-linear separations, known as the soft margin case.

### SVM Hard Margin Case
The boundary of a hard margin is explicit and constitutes the “hard margin” of a linear support vector machine, also known as *"Hard Margin SVM"*. The general idea is a dataset where “all” points must be linearly separable into its resulting class, that is one side and one side only has a single class of points, whilst another side also only has a single class of points. This case however, allows for no mixed class separation when fitting a support vector, where any point that lies on the incorrect side of a hyperplane may results in failing to classify points by prediction.

### SVM Soft Margin Case
On the other hand, the case of points lying on the incorrect side of a hyperplane can be solved by modifying the boundaries to reflect these changes amd allowing such points to be on either side, by applying a regulating parameter *<img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>*. This such case constitutes the “soft margin” of a non-linear support vector machine, also known as *"Soft Margin SVM"*. Consequently, the effect of such parameter "C" can vastly effect the accuracy of classifications and lead to the generalization problem, with whether a more accurate separator is deemed to be a better suite then a generalized separator.

That is to say, whether having a lower regulating parameter *<img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>* value with less outlier points correctly classified, proportional to a larger max-margin width is a better end classification model that works for unseen data. Or a highly tuned model tailored with a larger regulating parameter *<img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>*, with a inversely proportional max-margin width and high accuracy to seen data is deemed more important.

### SVM Algorithmic Implementation Details
Since the problems will be implemented in [CVXOPT](https://cvxopt.org/), the form for the dual need to be converted into canonical form as given by [quadratic programming form](https://cvxopt.org/userguide/coneprog.html#quadratic-programming) in CVXOPT api, such that the following are met:
<p align="center"><img src="/tex/3f685b8d4b0ddc36f878384d63577365.svg?invert_in_darkmode&sanitize=true" align=middle width=195.57880815pt height=78.0312258pt/></p>

#### SVM Dual Problem
> As a side note, since the dual problem form is derived from the primal problem, a short glimpse.. actually maybe a dive.. into the primal form is given below.
> The primal form of a hard margin in which is specified by:
> <p align="center"><img src="/tex/0917d5279d0653575375fa77c91354f9.svg?invert_in_darkmode&sanitize=true" align=middle width=280.40620905000003pt height=59.969357249999995pt/></p>
>
> Furthermore, since it is required to solve for two variables, <img src="/tex/ee31e753d9939d414e8314187a13b0fe.svg?invert_in_darkmode&sanitize=true" align=middle width=26.571525749999992pt height=22.831056599999986pt/>, such variables can be combined into a singular matrix, such that <img src="/tex/2176d0eef5ebd7c1a35826c72778bba7.svg?invert_in_darkmode&sanitize=true" align=middle width=72.70329659999999pt height=24.65753399999998pt/>. Also, since the primal form is already quite marginally similar to the CVXOPT canonical form, the inverse of the conditions can be taken to end up in the required CVXOPT form of:
> <p align="center"><img src="/tex/04731bdf29ece56abb908fef3d1b423c.svg?invert_in_darkmode&sanitize=true" align=middle width=313.28283359999995pt height=59.969357249999995pt/></p>
>
> In addition, given that it is intended to solve a soft margin primal problem, the regulating parameter *<img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>* can be added such with a slack variable *"<img src="/tex/85e60dfc14844168fd12baa5bfd2517d.svg?invert_in_darkmode&sanitize=true" align=middle width=7.94809454999999pt height=22.831056599999986pt/>"*, such that the new equation and constraints for a soft margin primal problem becomes:
> <p align="center"><img src="/tex/787e2e08c441000c23ad7855367fb274.svg?invert_in_darkmode&sanitize=true" align=middle width=346.03863359999997pt height=95.62074555pt/></p>

Derived from the primal problem, along with the lagrange multipliers <img src="/tex/3042c543b3b9c0941ca1d3eea87e8519.svg?invert_in_darkmode&sanitize=true" align=middle width=34.11721994999999pt height=24.65753399999998pt/>, the dual problem in the hard margin form is:
<p align="center"><img src="/tex/dbed248b71d9048562905b2874bb2af4.svg?invert_in_darkmode&sanitize=true" align=middle width=239.09643945pt height=47.1348339pt/></p>

<p align="center"><img src="/tex/b71befb48b362bf5fbc2e7d12afec0f3.svg?invert_in_darkmode&sanitize=true" align=middle width=175.13798774999998pt height=70.4499609pt/></p>

To convert the problem into an a solvable CVXOPT canonical form of <img src="/tex/6819b215dac1c2e8894599501ea59af8.svg?invert_in_darkmode&sanitize=true" align=middle width=103.75665794999999pt height=27.77565449999998pt/>, let <img src="/tex/32a3b96eadd81450623b8c91678b2672.svg?invert_in_darkmode&sanitize=true" align=middle width=28.323944549999993pt height=22.465723500000017pt/> to represent the matrix form of <img src="/tex/a132b81870a2ceaf0c7a4fe6ca2ab0c4.svg?invert_in_darkmode&sanitize=true" align=middle width=71.93687654999998pt height=27.15900329999998pt/> , the dual form hence becomes of <img src="/tex/de4ee44754783b8901e32565016352bc.svg?invert_in_darkmode&sanitize=true" align=middle width=193.63204244999997pt height=27.77565449999998pt/>. In addition, to obtain the required form, the removal of summations through the use of vectors and inverses of the whole equation and conditions, turn a maximize problem into a minimize problem and required CVXOPT canonical form of:
<p align="center"><img src="/tex/43c66ddfa6bc43030125ce404f765dfa.svg?invert_in_darkmode&sanitize=true" align=middle width=135.8694975pt height=32.990165999999995pt/></p>

<p align="center"><img src="/tex/c8f8bbb67bf34b7f352b8499c21fc33a.svg?invert_in_darkmode&sanitize=true" align=middle width=145.32432914999998pt height=41.75538345pt/></p>

From this, by directly mapping corresponding values to the canonical form, values obtained are:
- P is a matrix is dimensions corresponding to <img src="/tex/1e69bd7360709a73ebf7f80e82a2aba7.svg?invert_in_darkmode&sanitize=true" align=middle width=143.70387734999997pt height=37.80850590000001pt/>
- q is a matrix of -1 with same dimensions but vertically as a single column
- G is a identity matrix corresponding to the dimensions of <img src="/tex/0aae089ed20772138e327117bd8c6bac.svg?invert_in_darkmode&sanitize=true" align=middle width=13.340053649999989pt height=14.15524440000002pt/>
- h is a matrix of 0’s corresponding to the dimensions of <img src="/tex/0aae089ed20772138e327117bd8c6bac.svg?invert_in_darkmode&sanitize=true" align=middle width=13.340053649999989pt height=14.15524440000002pt/>
- A is a matrix of the labels position rotated horizontally
- b is a matrix containing a single <img src="/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>

Furthermore, using the above form of a hard margin, a soft margin primal problem can be derived where a regulating parameter *<img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>* is added,  such that another parameter of constraint is added to the conditions on <img src="/tex/0aae089ed20772138e327117bd8c6bac.svg?invert_in_darkmode&sanitize=true" align=middle width=13.340053649999989pt height=14.15524440000002pt/>, such that <img src="/tex/9da19e4ee589cbdeef89d96ec58efb5e.svg?invert_in_darkmode&sanitize=true" align=middle width=79.1410653pt height=22.465723500000017pt/>. Such that now the condition <img src="/tex/6ffb5786b0d27de0d37849a048c3f257.svg?invert_in_darkmode&sanitize=true" align=middle width=43.476892799999995pt height=21.18721440000001pt/> becomes <img src="/tex/98a0bffdf730c60555d6adfa97fab72b.svg?invert_in_darkmode&sanitize=true" align=middle width=57.08422499999998pt height=21.18721440000001pt/>. Hence the soft margin dual problem form becomes:
<p align="center"><img src="/tex/7a22632d5352a7c61304f31c973fad6b.svg?invert_in_darkmode&sanitize=true" align=middle width=135.8694975pt height=32.990165999999995pt/></p>

<p align="center"><img src="/tex/1e74c7c9e717dcc80cb6b2f11bd7a47b.svg?invert_in_darkmode&sanitize=true" align=middle width=151.88351805pt height=66.41291745pt/></p>

In addition, with such new constraint, the canonical form parameters are modified such that:
- G is now an negative identity matrix corresponding to the dimensions of <img src="/tex/0aae089ed20772138e327117bd8c6bac.svg?invert_in_darkmode&sanitize=true" align=middle width=13.340053649999989pt height=14.15524440000002pt/> stacked below with another identity <img src="/tex/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode&sanitize=true" align=middle width=8.515988249999989pt height=22.465723500000017pt/> matrix corresponding to the dimensions of <img src="/tex/0aae089ed20772138e327117bd8c6bac.svg?invert_in_darkmode&sanitize=true" align=middle width=13.340053649999989pt height=14.15524440000002pt/>
- h is now an matrix of <img src="/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>’s corresponding to the dimensions of <img src="/tex/0aae089ed20772138e327117bd8c6bac.svg?invert_in_darkmode&sanitize=true" align=middle width=13.340053649999989pt height=14.15524440000002pt/> stacked horizontally below with another matrix of one multiplied by regulating parameter constant *<img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>*.

##### Calculating Weights and Bias in The Dual Problem
By solving this quadratic programming problem, the obtained results only resemble such lagrange mulitpliers <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/>. But to solve for <img src="/tex/ee31e753d9939d414e8314187a13b0fe.svg?invert_in_darkmode&sanitize=true" align=middle width=26.571525749999992pt height=22.831056599999986pt/> the formula is constructed based of the derivation from the lagrange function of:
<p align="center"><img src="/tex/71529bbbf8c86919f6d281aad6879b56.svg?invert_in_darkmode&sanitize=true" align=middle width=326.23144455pt height=44.89738935pt/></p>

By deriving, the obtained derivation for <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/>:
<p align="center"><img src="/tex/516f2c455a11f4006082474ec0e2d43a.svg?invert_in_darkmode&sanitize=true" align=middle width=284.43159525pt height=44.89738935pt/></p>

and by rearranging such equation, the formula for obtaining the value of <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/> becomes:
<p align="center"><img src="/tex/39e97504fd1bd28650cd0a01cedf7e24.svg?invert_in_darkmode&sanitize=true" align=middle width=140.13398684999999pt height=44.89738935pt/></p>

On the other hand, we can obtain the value for <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> by using the idea that <img src="/tex/6eee1f4d8ad7c1be15049dae8dbeee5f.svg?invert_in_darkmode&sanitize=true" align=middle width=171.42267285pt height=24.65753399999998pt/>, where <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/> gets cancelled out due to points being close to 0, leaving the form <img src="/tex/2abd5946c013f893c44b3a97dc0d9200.svg?invert_in_darkmode&sanitize=true" align=middle width=145.38870555pt height=24.65753399999998pt/>, where by cancelling each side by <img src="/tex/9691fd7a74a8c4366fff6c5aa4960dbe.svg?invert_in_darkmode&sanitize=true" align=middle width=85.26333419999999pt height=26.76175259999998pt/>, remains with <img src="/tex/a10a07149b9cb78de1eb4d24d7fdd417.svg?invert_in_darkmode&sanitize=true" align=middle width=124.38406199999997pt height=22.831056599999986pt/>. Rearranging such equation with <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> as the subject leaves (where <img src="/tex/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode&sanitize=true" align=middle width=11.027402099999989pt height=22.465723500000017pt/> is the support vector):
<p align="center"><img src="/tex/96effda704226187e544c56a61873189.svg?invert_in_darkmode&sanitize=true" align=middle width=129.23439374999998pt height=16.438356pt/></p>

##### Classifying Points in The Dual Problem
Since a support vector machine is of a linear separable hyperplane, which is of the form <img src="/tex/0419db7b2713b845f175c8ac5802eb19.svg?invert_in_darkmode&sanitize=true" align=middle width=75.79696739999999pt height=22.831056599999986pt/>, once an hyperplane plane is obtained consisting of <img src="/tex/31dd81634e85e6307a1ad98007005174.svg?invert_in_darkmode&sanitize=true" align=middle width=26.571525749999992pt height=22.831056599999986pt/>, the form of <img src="/tex/927334713b2d326542ab6a748f1da9b2.svg?invert_in_darkmode&sanitize=true" align=middle width=90.76064414999998pt height=22.831056599999986pt/>, which is the formula used to calculate the prediction of a classified point. In addition, the class which the prediction falls under is the idea that if we take the sign of the outcome. Hence, then the prediction becomes:
<p align="center"><img src="/tex/8c3d7cce9c1daadb13535287734ae37c.svg?invert_in_darkmode&sanitize=true" align=middle width=182.61758625pt height=49.315569599999996pt/></p>

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

Since the Wisconsin Diagnostic Breast Cancer dataset consisted of 569 samples, the optimal separation of data into a training and testing set with a balance between accuracy and precision was 250 training samples (319 testing samples). Through testing, it was found that training such predictive model below 250 samples, would result with the accuracy of both the training and testing set would be substantially lower then a predictive model using only Support Vector Machines (SVM) as its basis. On the other hand, although training with 250 - 300 samples would produce accurate results with a fluctuation of <img src="/tex/25cfe28e7768fb30f094b4bda52db52a.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=21.18721440000001pt/> between the highest and lowest accuracy of between both sets, any trained samples greater then the specified optimal range resulted in an over fitted model. where the accuracy of the training set would be significantly higher in accuracy with values in mid to high 90's, whilst the testing data would suffer in accuracy resulting in high 80's to low 90's.

Based on such model trained with 250 - 300 samples and varying parameter regulations <img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>, as the regulating parameter <img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/> was increased, with the SVM margin being more tailored to the training set, it was seen that there was a higher degree in accuracy with training reaching a maximum threshold of <img src="/tex/4b98de734f59e61811dcd1b5f1f19f86.svg?invert_in_darkmode&sanitize=true" align=middle width=42.922524149999994pt height=24.65753399999998pt/>. Whilst, on the other hand, the testing set resulted in a significantly higher error rate then a normal Support Vector Machine model. On the contrary, with a lower regulating parameter <img src="/tex/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/>, both sets had a marginally higher error rate when compared to a SVM model.

To Determine and choose the optimal model to represent the Wisconsin Diagnostic Breast Cancer dataset, a model was chosen in which the following where met, with highest priority being listed first:
- A AdaBoost-SVM model where both sets achieve an accuracy greater then a model using only Support Vector Machine.
- A AdaBoost-SVM model where the fluctuations in errors between both sets are as minimal and as close as possible.
- A Model with a gradual decrement (Not instant) from initial Support Vector Machine model, to a error/accuracy theoretical maxima.

Hence, from these conclusions the predictive model with **Optimal Model Parameters** which was used to predict the diagnosis of breast cancer cells as either Benign or Malignant was as follows:
```bash
python -m adaboost dataset_file=default_1 dataset_sample_size=250 svm_regularizer_c=10 adaboost_estimators=8
```

This obtained model, achieved a 5% error rate on average between both sets, with both sets having the smallest deviation margin difference in error, whilst still obtaining a high classification rate in the mid <img src="/tex/cb2597957123c407ff32edcb5c76b7ce.svg?invert_in_darkmode&sanitize=true" align=middle width=30.137091599999987pt height=24.65753399999998pt/>, as shown below in Figure 1.

**Error Loss & Accuracy:**
<img src="/figs/plot/default_1_ss250_c10_ae8.png" width=750 height=450/>

*Figure 1: Graph showing error loss and model accuracy against number of iterators, using specified model parameters above.*

As a note, Since the dataset is rather small in terms of features, the amount of used weak-learners was also rather significantly smaller than initially thought. In addition, this may have also been effected as a Support Vector Machine classifier is rather an already *strong* classifier itself, with tendency to have higher then accuracy starting in the <img src="/tex/ba566c2176e6618f6ed5a7c6928a48e8.svg?invert_in_darkmode&sanitize=true" align=middle width=30.137091599999987pt height=24.65753399999998pt/>. Whilst, AdaBoost tends works better in generating predictive models with weaker classifiers that obtain <img src="/tex/f84dc7a2c2fe4b38a41e23af3644b3d2.svg?invert_in_darkmode&sanitize=true" align=middle width=80.36537354999999pt height=24.65753399999998pt/> accuracy.
Although, generally an AdaBoost SVM model is trained using a RBF Kernel-SVM, this may be implemented in a future stage to see if the model can achieve a more stable and preforming model for future unseen cases.
