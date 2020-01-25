import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from adaboost import methods
from adaboost.common import constants, classes, classification, format_dataset
from adaboost.common.check import check_type
from adaboost.common.get import retrieve_param
from adaboost.common.set import set_param
from adaboost.learning.dimension_reduction import pca
from adaboost.learning.weak_learner.classifier import svm

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

  # Initialize global variables
  constants.initialize()


#####################
# START APPLICATION #
#####################

# ============================================
# Retrieve all parameter values for initlizing
# ============================================

  # Extract parameter values from user input or application arguments
  if len(sys.argv) > 1:
    print("Retrieving parameters from arguments...")
    raw_dataset = retrieve_param.arg_dataset()
    if raw_dataset != "default":
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
    if raw_dataset != "default":
      raw_dataset_label_col = retrieve_param.input_dataset_label_col()
      raw_dataset_feature_cols = retrieve_param.input_dataset_features_col()
    raw_dataset_sample_size = retrieve_param.input_dataset_sample_size()
    raw_pca_reduction = retrieve_param.input_pca_reduction()
    raw_svm_regularizer_c = retrieve_param.input_svm_regularizer_c()
    raw_adaboost_estimators = retrieve_param.input_adaboost_estimators()
    raw_out_detail = retrieve_param.input_out_detail()

  # Ensure all parameters values retrieved have a valid field
  if raw_dataset != "default":
    retrieve_col_none_check = (
      raw_dataset_label_col is None
      or raw_dataset_feature_cols is None
    )
    if retrieve_col_none_check:
      print("Exiting. (ERR_UNSET_RAW_PARAM_COL)")
      sys.exit(-1)

  retrieve_none_check = (
    raw_dataset is None
    or raw_dataset_sample_size is None
    or raw_pca_reduction is None
    or raw_adaboost_estimators is None
    or raw_out_detail is None
  )
  if retrieve_none_check:
    print("Exiting. (ERR_UNSET_RAW_PARAM)")
    sys.exit(-1)

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
    if raw_dataset == 'default':
      if constants.OUTPUT_DETAIL is True:
        print("  --set Dataset-Label-Column: 1")
        print("  --set Dataset-Feature-Column: [2 - 32]")
      # convert labels in default WDBC_dataset from string [M, B] to int [1, -1]
      DATASET = format_dataset.dataset_default_label(DATASET_FILEPATH)
      dataset_label_col = 1
      dataset_feature_start_col = 2
      dataset_feature_end_col = 32
    else:
      DATASET = pd.read_csv(DATASET_FILEPATH, header=None)
      dataset_label_col = set_param.dataset_label_col(raw_dataset_label_col, DATASET.shape[1])
      dataset_feature_start_col, dataset_feature_end_col = set_param.dataset_feature_cols(raw_dataset_feature_cols, DATASET.shape[1])

    # Application initializer check to ensure all column parameters have been set
    set_column_none_check = (
      dataset_label_col is None
      or dataset_feature_start_col is None
      or dataset_feature_end_col is None
    )
    if set_column_none_check:
      print("Exiting. (ERR_SET_PARAM_COL)")
      sys.exit(-1)

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
  set_none_check = (
    DATASET_FILEPATH is None
    or dataset_sample_size is None
    or dataset_test_size is None
    or dataset_feature_count is None
    or pca_reduction is None
    or svm_regularizer_c is None
    or adaboost_estimators is None
  )
  if set_none_check:
    print("Exiting. (ERR_SET_PARAM)")
    sys.exit(-1)


# ================================================================================
# Print out all initialized parameter values as a confirmation everthing succeeded
# ================================================================================

  print("\n=== Initialized Parameter Details ===")
  print("> Dataset: %s\n"
          "\t- Malignant Label: %s\n"
          "\t- Benign Label: %s\n"
          "\t- Feature Columns: [%s] - [%s]  (%s Features)\n"
          "\t- Sample Size: %s\n"
          "\t- Test Size: %s\n\n"
        "> PCA:\n"
          "\t- Reduction Size: %s\n\n"
        "> SVM:\n"
          "\t- Regularizer C: %s\n\n"
        "> AdaBoost:\n"
          "\t- Estimators: %s"
        % (DATASET_FILEPATH.rsplit('/', 1)[-1], constants.MALIGNANT_LABEL, constants.BENIGN_LABEL,
          dataset_feature_start_col, dataset_feature_end_col, dataset_feature_count,
          dataset_sample_size, dataset_test_size,
          pca_reduction,
          svm_regularizer_c,
          adaboost_estimators))


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
      sys.exit(-1)

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

  for estimators in range(adaboost_estimators):
    print("> AdaBoost Estimator :: [ %d ]" % (estimators + 1))
    print("--------------------------------")

    # Weak Learner - SVM application
    hypothesis = classes.model()
    svm_w, svm_b = svm.application.run (
                      dataset_train_set,
                      dataset_train_label,
                      svm_regularizer_c,
                      adaboost_distribution_weights
                    )

    # Set SVM hypothesis parameters
    hypothesis = methods.set_hypothesis_value(hypothesis, svm_w, svm_b)

    # Predict classification errors in labels (y) with the relationship X -> y
    hypothesis_prediction, hypothesis_error = methods.hypothesis_classification_error (
                                                hypothesis,
                                                dataset_train_set, dataset_train_label,
                                                adaboost_distribution_weights
                                              )

    # Calculate significance of hypothesis model as a final strong learner
    hypothesis.model_weight = methods.hypothesis_significance(hypothesis_error)

    # Update distribution weights
    adaboost_distribution_weights = methods.update_distribution_weights (
                                      hypothesis.model_weight, hypothesis_prediction,
                                      dataset_train_label,
                                      adaboost_distribution_weights
                                    )

    print("\n  AdaBoost Model Weight: [%s]\n"
          "================================"
          % (hypothesis.model_weight))
    if estimators != (adaboost_estimators - 1):
      print("\n")

    if hypothesis.model_weight == 0.0:
      adaboost_early_termination = True
      break

    adaboost_model.append(hypothesis)

  if adaboost_early_termination is True:
    print("Early Termination. Stopping. (EARLY_TERMINATION_DETECTED)")


  hypothesis_classification = methods.hypothesis_final(adaboost_model, dataset_train_set, dataset_sample_size)

  hypothesis_classification1 = methods.hypothesis_final(adaboost_model, dataset_test_set, dataset_test_size)

  test = classification.prediction_accuracy(hypothesis_classification, dataset_train_label)
  test = classification.prediction_accuracy(hypothesis_classification1, dataset_test_label)



  sys.exit(0)
# # Predict label of corresponding dataset using: label = wx + b
# def svm_classification_prediction(dataset, w, b):
#   return np.sign(np.dot(dataset, w) + b[0])


# def svm_classification_prediction_accuracy(dataset_labels, prediction):
#   return np.hstack((prediction == dataset_labels))