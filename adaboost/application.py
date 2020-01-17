import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from adaboost.common import constants, format_dataset
from adaboost.common.check import check_type
from adaboost.common.get import retrieve_param
from adaboost.common.set import set_param
from adaboost.learning.dimension_reduction import pca

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
  raw_adaboost_estimators = None
  raw_out_detail = None

  # Dataset data
  DATASET = None
  DATASET_FILEPATH = None
  dataset_label_col = None
  datset_feature_count = None
  datset_feature_start_col = None
  datset_feature_end_col = None

  # Training set data
  dataset_sample_size = None
  dataset_train_set = None
  dataset_train_label = None

  # Testing set data
  dataset_test_size = None
  dataset_test_set = None
  dataset_test_label = None

  # PCA parameters
  pca_reduction = None

  # AdaBoost parameters
  adaboost_estimators = None

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
  # also partition datset into corresponding training and testing set
  DATASET_FILEPATH = set_param.dataset(raw_dataset)
  if DATASET_FILEPATH is not None:
    # If default dataset specified, pre-initialize with following parameter values for:
    # dataset_label_col, dataset_feature_start_col, dataset_feature_end_col
    if raw_dataset == 'default':
      if constants.OUTPUT_DETAIL is True:
        print("--set Dataset-Label-Column: 1")
        print("--set Dataset-Feature-Column: [2 - 32]")
      # convert labels in default WDBC_dataset from string [M, B] to int [1, -1]
      DATASET = format_dataset.dataset_labels(DATASET_FILEPATH)
      dataset_label_col = 1
      datset_feature_start_col = 2
      datset_feature_end_col = 32
    else:
      DATASET = pd.read_csv(DATASET_FILEPATH, header=None)

    # Partition dataset into training and testing set
    dataset_sample_size, dataset_test_size = set_param.dataset_sample_test_size(raw_dataset_sample_size, DATASET.shape[0])
    if check_type.is_int(dataset_sample_size) and check_type.is_int(dataset_test_size):
      if constants.OUTPUT_DETAIL is True:
        print("--set -init Dataset-Train-Set")
        print("--set -init Dataset-Test-Set")
      dataset_train_set = DATASET.head(dataset_sample_size)
      dataset_test_set = DATASET.tail(dataset_test_size)

    # If dataset was specified, initialize the following parameter values from retrieved values:
    # dataset_label_col, dataset_feature_start_col, dataset_feature_end_col
    if raw_dataset != 'default':
      dataset_label_col = set_param.dataset_label_col(raw_dataset_label_col, DATASET.shape[1])
      datset_feature_start_col, datset_feature_end_col = set_param.dataset_feature_cols(raw_dataset_feature_cols, DATASET.shape[1])

    # Application initializer check to ensure all column parameters have been set
    set_column_none_check = (
      dataset_label_col is None
      or datset_feature_start_col is None
      or datset_feature_end_col is None
    )
    if set_column_none_check:
      print("Exiting. (ERR_SET_PARAM_COL)")
      sys.exit(-1)

    dataset_test_label = format_dataset.extract_columns(dataset_test_set, dataset_label_col)
    dataset_train_label = format_dataset.extract_columns(dataset_train_set, dataset_label_col)
    datset_feature_count = datset_feature_end_col - datset_feature_start_col

    pca_reduction = set_param.pca_reduction(raw_pca_reduction, datset_feature_count)
    adaboost_estimators = set_param.adaboost_estimators(raw_adaboost_estimators)

  # Application initializer check to ensure last initializers have been set and are not None
  set_none_check = (
    DATASET_FILEPATH is None
    or dataset_sample_size is None
    or dataset_test_size is None
    or datset_feature_count is None
    or pca_reduction is None
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
        "> AdaBoost:\n"
          "\t- Estimators: %s"
        % (DATASET_FILEPATH.rsplit('/', 1)[-1], constants.MALIGNANT_LABEL, constants.BENIGN_LABEL,
          datset_feature_start_col, datset_feature_end_col, datset_feature_count,
          dataset_sample_size, dataset_test_size,
          pca_reduction,
          adaboost_estimators))


# ======================
# PCA application starts
# ======================

  # Execute dimensionality reduction on dataset by using PCA
  if pca_reduction != 'none':
    if constants.OUTPUT_DETAIL is True:
      print("\n=== PCA Initialize Details ===")

    reduction_value = pca.application.run (
                        pca_reduction,
                        dataset_train_set,
                        dataset_sample_size,
                        datset_feature_count,
                        datset_feature_start_col,
                        datset_feature_end_col
                      )
    if reduction_value is None:
      print("Exiting. (ERR_MIN_THRESHOLD)")
      sys.exit(-1)

    feature_order = reduction_value[2]
    if constants.OUTPUT_DETAIL is True:
      print("--set -update Dataset-Feature-Count")
      print("--set -update Dataset-Train-Set")
      print("--set -update Dataset-Test-Set")

    dataset_feature_count = reduction_value[0]
    dataset_train_set = reduction_value[1]
    dataset_test_set = format_dataset.extract_columns(dataset_test_set, feature_order + datset_feature_start_col)
  else:
    if constants.OUTPUT_DETAIL is True:
      print("\n--skipped PCA")

  sys.exit(0)
