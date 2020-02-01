import math
import numpy as np
import pandas as pd
import sys
import time
from adaboost import methods
from adaboost.common import constants, classes, classification, format_dataset
from adaboost.common.check import check_application, check_type
from adaboost.common.get import application_helper, retrieve_param
from adaboost.common.set import set_param
from adaboost.learning.dimension_reduction import pca
from adaboost.learning.weak_learner.classifier import svm
from adaboost.plotting import plot_application

def run():
# ============================================
# Initlizing default values
# ============================================

  # Data parameters
  raw_dataset = None
  raw_dataset_label_col = None
  raw_dataset_feature_cols = None
  raw_dataset_sample_size = None
  raw_pca_reduction = None
  raw_svm_regularizer_c = None
  raw_adaboost_estimators = None
  raw_out_detail = None

  # Dataset data
  DATASET = None
  DATASET_FILEPATH = None
  dataset_label_col = None
  dataset_labels = None
  dataset_feature_count = None
  dataset_feature_start_col = None
  dataset_feature_end_col = None

  # Training set data
  dataset_sample_size = None
  dataset_train_set = None
  dataset_train_label = None

  # Testing set data
  dataset_test_size = None
  dataset_test_set = None
  dataset_test_label = None

  # PCA data
  pca_reduction = None

  # SVM data
  svm_regularizer_c = None

  # AdaBoost data
  adaboost_estimators = None
  adaboost_dataset_weights = None
  adaboost_model = []
  adaboost_early_termination = False
  adaboost_error_termination = False

  # Application plotting data
  application_train_error = []
  application_train_accuracy = []
  application_test_error = []
  application_test_accuracy = []

  # Initialize global variables
  constants.initialize()


#####################
# START APPLICATION #
#####################

# ============================================
# Retrieve all parameter values for initlizing
# ============================================

  # Extract parameter values from user input or application arguments
  if len(sys.argv) > 2:

    print("Retrieving parameters from arguments...")

    raw_dataset = retrieve_param.arg_dataset()
    if not check_application.default_datasets(raw_dataset):
      raw_dataset_label_col = retrieve_param.arg_dataset_label_col()
      raw_dataset_feature_cols = retrieve_param.arg_dataset_features_col()
    raw_dataset_sample_size = retrieve_param.arg_dataset_sample_size()
    raw_pca_reduction = retrieve_param.arg_pca_reduction()
    raw_svm_regularizer_c = retrieve_param.arg_svm_regularizer_c()
    raw_adaboost_estimators = retrieve_param.arg_adaboost_estimators()
    raw_out_detail = retrieve_param.arg_out_detail()
  else:
    print("Retrieving parameters from user input...")
    raw_dataset = retrieve_param.input_dataset()
    if not check_application.default_datasets(raw_dataset):
      raw_dataset_label_col = retrieve_param.input_dataset_label_col()
      raw_dataset_feature_cols = retrieve_param.input_dataset_features_col()
    raw_dataset_sample_size = retrieve_param.input_dataset_sample_size()
    raw_pca_reduction = retrieve_param.input_pca_reduction()
    raw_svm_regularizer_c = retrieve_param.input_svm_regularizer_c()
    raw_adaboost_estimators = retrieve_param.input_adaboost_estimators()
    raw_out_detail = retrieve_param.input_out_detail()

  # Ensure all parameters values retrieved have a valid field
  if not check_application.default_datasets(raw_dataset):
    if check_application.none_check ([
      raw_dataset_label_col, raw_dataset_feature_cols
    ]):
      print("Exiting. (ERR_UNSET_RAW_PARAM_COL)")
      return -1

  if check_application.none_check ([
    raw_dataset, raw_dataset_sample_size, raw_pca_reduction,
    raw_svm_regularizer_c, raw_adaboost_estimators, raw_out_detail
  ]):
    print("Exiting. (ERR_UNSET_RAW_PARAM)")
    return -1

  # Initialize application variables with extracted parameter values
  if constants.OUTPUT_DETAIL is True:
    print("Initialize:")


# ==========================================
# Initlizing values with obtained parameters
# ==========================================

  set_param.out_detail(raw_out_detail)

  # If dataset file exists, set and initialize dataset as a pandas dataframe,
  # also partition dataset into corresponding training and testing set
  DATASET_FILEPATH = set_param.dataset(raw_dataset)
  if DATASET_FILEPATH is not None:
    # If default dataset is specified, pre-initialize parameter values,
    # else use user specified parameter values
    if raw_dataset == 'default_1':
      if constants.OUTPUT_DETAIL is True:
        print("  --set Dataset-Label-Column: 1")
        print("  --set Dataset-Feature-Column: [2 - 32]")

      dataset_label_col = constants.WDBC_DATASET_LABEL_COL
      dataset_feature_start_col = constants.WDBC_DATASET_FEATURE_START_COL
      dataset_feature_end_col = constants.WDBC_DATASET_FEATURE_END_COL
    elif raw_dataset == 'default_2' or raw_dataset == 'default_3':
      if constants.OUTPUT_DETAIL is True:
        print("  --set Dataset-Label-Column: 0")
        print("  --set Dataset-Feature-Column: [1 - 201]")

      dataset_label_col = constants.SHEN_DATASET_LABEL_COL
      dataset_feature_start_col = constants.SHEN_DATASET_FEATURE_START_COL
      dataset_feature_end_col = constants.SHEN_DATASET_FEATURE_END_COL
    else:
      DATASET = pd.read_csv(DATASET_FILEPATH, header=None)
      dataset_label_col = set_param.dataset_label_col(raw_dataset_label_col, DATASET.shape[1])
      dataset_feature_start_col, dataset_feature_end_col = set_param.dataset_feature_cols(raw_dataset_feature_cols, DATASET.shape[1])
    DATASET, dataset_labels = format_dataset.dataset_default_label(DATASET_FILEPATH, dataset_label_col)

    # Application initializer check to ensure all column parameters have been set
    if check_application.none_check ([
      DATASET, dataset_label_col,
      dataset_feature_start_col, dataset_feature_end_col
    ]):
      print("Exiting. (ERR_SET_DATASET_PARAM_COL)")
      return -1

    # Update count for total number of features
    dataset_feature_count = dataset_feature_end_col - dataset_feature_start_col

    # Partition dataset into training and testing set
    dataset_sample_size, dataset_test_size = set_param.dataset_sample_test_size(raw_dataset_sample_size, DATASET.shape[0])
    if check_type.is_int(dataset_sample_size) and check_type.is_int(dataset_test_size):
      if constants.OUTPUT_DETAIL is True:
        print("  --set -init Dataset-Train-Set")
        print("  --set -init Dataset-Test-Set")

      dataset_train_set = DATASET.head(dataset_sample_size)
      dataset_train_label = format_dataset.dataset_extract_columns(dataset_train_set, dataset_label_col)
      dataset_train_set = format_dataset.dataset_extract_features (
                            dataset_train_set,
                            dataset_feature_start_col,
                            dataset_feature_end_col
                          )

      dataset_test_set = DATASET.tail(dataset_test_size)
      dataset_test_label = format_dataset.dataset_extract_columns(dataset_test_set, dataset_label_col)
      dataset_test_set = format_dataset.dataset_extract_features (
                            dataset_test_set,
                            dataset_feature_start_col,
                            dataset_feature_end_col
                          )

    pca_reduction = set_param.pca_reduction(raw_pca_reduction, dataset_feature_count)
    svm_regularizer_c = set_param.svm_regularizer_c(raw_svm_regularizer_c)
    adaboost_estimators = set_param.adaboost_estimators(raw_adaboost_estimators)

  # Application initializer check to ensure last initializers have been set and are not None
  if check_application.none_check ([
   DATASET_FILEPATH, dataset_test_size, dataset_feature_count,
    pca_reduction, svm_regularizer_c, adaboost_estimators
  ]):
    print("Exiting. (ERR_SET_PARAM)")
    return -1


# ================================================================================
# Print out all initialized parameter values as a confirmation everthing succeeded
# ================================================================================

  print("=== Initialized Parameter Details ===")
  print("> Dataset: %s\n"
          "\t- [%s] Label: %s\n"
          "\t- [%s] Label: %s\n"
          "\t- Feature Columns: [%s] - [%s]  (%s Features)\n"
          "\t- Sample Size: %s\n"
          "\t- Test Size: %s\n\n"
        "> PCA:\n"
          "\t- Reduction Size: %s\n\n"
        "> SVM:\n"
          "\t- Regularizer C: %s\n\n"
        "> AdaBoost:\n"
          "\t- Estimators: %s"
        % (DATASET_FILEPATH.rsplit('/', 1)[-1],
          dataset_labels[0][0], dataset_labels[0][1],
          dataset_labels[1][0], dataset_labels[1][1],
          dataset_feature_start_col, dataset_feature_end_col, dataset_feature_count,
          dataset_sample_size, dataset_test_size,
          pca_reduction,
          svm_regularizer_c,
          adaboost_estimators))

  application_start_time = time.time()


# ===============
# PCA application
# ===============

  # Execute dimensionality reduction on dataset by using PCA
  if pca_reduction != 'none':
    print("\n=== PCA Initialize Details ===")

    reduction_value = pca.application.run (
                        pca_reduction,
                        dataset_train_set,
                        dataset_sample_size,
                        dataset_feature_count
                      )

    if reduction_value is None:
      print("Exiting. (ERR_MIN_THRESHOLD)")
      return -1

    print("  --set -update Dataset-Feature-Count: [%d] -> [%s]"
          % (dataset_feature_count, reduction_value[2]))
    print("  --set -update Dataset-Train-Set: %s -> %s"
          % (dataset_train_set.shape, reduction_value[0].shape))
    print("  --set -update Dataset-Test-Set: %s -> (%s, %s)"
          % (dataset_test_set.shape, dataset_test_set.shape[0], reduction_value[2]))

    feature_order = reduction_value[1]
    dataset_feature_count = reduction_value[2]
    dataset_train_set = format_dataset.pandas_dataframe(reduction_value[0])
    dataset_test_set = format_dataset.dataset_extract_columns(dataset_test_set, feature_order)
  else:
    if constants.OUTPUT_DETAIL is True:
      print("\n--skipped PCA")


# ====================
# AdaBoost application
# ====================

  print("\n=== AdaBoost Runtime Details ===")

  # Initialize distribution weights
  adaboost_distribution_weights = methods.distribution_weights(dataset_sample_size)

  for estimators in range(adaboost_estimators + 1):
    estimator_start_time = time.time()

    print("> AdaBoost Estimator :: [ %d ]" % (estimators))
    print("--------------------------------------------------------")

    # Weak Learner - SVM application
    hypothesis = classes.model()

    # Run first (0th) iteration as SVM without AdaBoost weights
    if estimators > 0:
      svm_w, svm_b = svm.application.run (
                        dataset_train_set,
                        dataset_train_label,
                        svm_regularizer_c,
                        adaboost_distribution_weights
                      )
    else:
      svm_w, svm_b = svm.application.run (
                        dataset_train_set,
                        dataset_train_label,
                        svm_regularizer_c,
                        None
                      )

    # Check for empty SVM values
    if svm_w is None and svm_b is None:
      adaboost_error_termination = True
      break

    # Set SVM hypothesis parameters
    hypothesis = methods.set_hypothesis_value(hypothesis, svm_w, svm_b)

    if estimators > 0:
      # Predict classification errors in labels (y) with the relationship X -> y
      hypothesis_prediction, hypothesis_error = methods.hypothesis_classification_error (
                                                  hypothesis,
                                                  dataset_train_set, dataset_train_label,
                                                  adaboost_distribution_weights
                                                )

      if math.isclose(hypothesis_error, 0.0, abs_tol = 0.0):
        print ("========================================================\n"
              "\nReached prediction error threshold. (classification error: 0.0).")

        adaboost_early_termination = True
        adaboost_estimators = estimators - 1
        break

      # Calculate significance of hypothesis model as a final strong learner
      hypothesis.model_weight = methods.hypothesis_significance(hypothesis_error)

      # Update distribution weights
      adaboost_distribution_weights = methods.update_distribution_weights (
                                        hypothesis.model_weight, hypothesis_prediction,
                                        dataset_train_label,
                                        adaboost_distribution_weights
                                      )

      print("\n  AdaBoost Model Weight: [%s]\n" % (hypothesis.model_weight))

      # Check weak learners contain only classifying information
      if math.isclose(abs(hypothesis.model_weight), 0.0, abs_tol = 1e-4):
        adaboost_early_termination = True
        adaboost_estimators = estimators - 1
        break
    else:
       hypothesis.model_weight = -1

    adaboost_model.append(hypothesis)

    estimator_end_time = time.time()

    # Obtain Prediction and accuracy using obtained AdaboostSVM model
    train_prediction = methods.hypothesis_final(adaboost_model, dataset_train_set, dataset_sample_size)
    train_results = classification.prediction_accuracy(train_prediction, dataset_train_label)
    test_prediction = methods.hypothesis_final(adaboost_model, dataset_test_set, dataset_test_size)
    test_results = classification.prediction_accuracy(test_prediction, dataset_test_label)

    application_train_error.append(train_results.classified_error)
    application_train_accuracy.append(train_results.classified_accuracy)
    application_test_error.append(test_results.classified_error)
    application_test_accuracy.append(test_results.classified_accuracy)

  # ===================
  # Application Results
  # ===================

    print("\n  > Application Result Details")
    print("  Estimator Execution Time: %s sec\n\n"
          "\n  - Train Set [%s samples]\n"
          "\t    Correctly Classified Samples: %s\n"
          "\t    Misclassified Samples: %s\n"
          "\t    Accuracy: %f%s\n"
          "\t    Error: %f%s\n"
          "\n  - Test Set [%s samples]\n"
          "\t    Correctly Classified Samples: %s\n"
          "\t    Misclassified Samples: %s\n"
          "\t    Accuracy: %f%s\n"
          "\t    Error: %f%s"
          % (estimator_end_time - estimator_start_time,
          dataset_sample_size, train_results.classified, train_results.misclassified,
          train_results.classified_accuracy * 100, '%',  train_results.classified_error * 100, '%',
          dataset_test_size, test_results.classified, test_results.misclassified,
          test_results.classified_accuracy * 100, '%', test_results.classified_error * 100, '%'))
    print("========================================================")
    if estimators != (adaboost_estimators - 1):
      print("\n")

  application_end_time = time.time()

  if adaboost_error_termination is True:
    print("Exiting. (ERR_SVM_NULL_PARAM)")
    return -1

  if adaboost_early_termination is True:
    print("Early termination. Stopping. (EARLY_TERMINATION_DETECTED)")

  print("Application Execution Time: %s sec" % (application_end_time - application_start_time))


  # ==========================
  # Plot Graphical Performance
  # ==========================

  if adaboost_estimators > 0:
    plot_application.run (
      adaboost_estimators,
      application_train_error, application_test_error,
      application_train_accuracy, application_test_accuracy
    )
  else:
    print("No model with weak learners found. Stopping. (NULL_PREDICTION_MODEL)")

  return 0
