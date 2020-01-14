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
  DATASET = None
  DATASET_FILEPATH = None
  datset_feature_count = None
  datset_feature_start_col = None
  datset_feature_end_col = None
  dataset_sample_size = None
  dataset_test_size = None
  dataset_train_set = None
  dataset_test_set = None
  pca_reduction = None
  adaboost_estimators = None

  raw_dataset = None
  raw_dataset_sample_size = None
  raw_pca_reduction = None
  raw_adaboost_estimators = None
  raw_out_detail = None

  # Initialize global variables
  constants.initialize()

  # Extract parameter values from user input or application arguments
  if len(sys.argv) > 1:
    print("Retrieving parameters from arguments...")
    raw_dataset = retrieve_param.arg_dataset()
    raw_dataset_sample_size = retrieve_param.arg_dataset_sample_size()
    raw_pca_reduction = retrieve_param.pca_arg_reduction()
    raw_adaboost_estimators = retrieve_param.adaboost_arg_estimators()
    raw_out_detail = retrieve_param.arg_out_detail()
  else:
    print("Retrieving parameters from user input...")
    raw_dataset = retrieve_param.input_dataset()
    raw_dataset_sample_size = retrieve_param.input_dataset_sample_size()
    raw_pca_reduction = retrieve_param.pca_input_reduction()
    raw_adaboost_estimators = retrieve_param.adaboost_input_estimators()
    raw_out_detail = retrieve_param.input_out_detail()

  # Ensure all parameters have a valid field
  retrieve_none_check = (
    raw_dataset is None
    or raw_dataset_sample_size is None
    or raw_pca_reduction is None
    or raw_adaboost_estimators is None
    or raw_out_detail is None
  )
  if retrieve_none_check:
    print("Exiting. (ERR_UNSET_PARAM)")
    sys.exit(-1)

  # Initialize application variables with extracted parameter values
  if constants.OUTPUT_DETAIL is True:
    print("Initialize:")

  set_param.out_detail(raw_out_detail)

  # If dataset file exists, set and initialize dataset as a pandas dataframe,
  # also partition datset into corresponding training and testing set
  DATASET_FILEPATH = set_param.dataset(raw_dataset)
  if DATASET_FILEPATH is not None:
    # If dataset is default, convert labels in WDBC_dataset from string to int
    if raw_dataset == 'default':
      if constants.OUTPUT_DETAIL is True:
        print("--set Dataset-Labels")
        print("--set Dataset-Feature-Start-Column")
        print("--set Dataset-Feature-End-Column")
      DATASET = format_dataset.dataset_labels(DATASET_FILEPATH)
      datset_feature_start_col = 2
      datset_feature_end_col = 32
      datset_feature_count = datset_feature_end_col - datset_feature_start_col

    # Partition dataset into training and testing set
    dataset_sample_size, dataset_test_size = set_param.dataset_sample_test_size(raw_dataset_sample_size, DATASET.shape[0])
    if check_type.is_int(dataset_sample_size) and check_type.is_int(dataset_test_size):
      if constants.OUTPUT_DETAIL is True:
        print("--init Dataset-Train-Set")
        print("--init Dataset-Test-Set")
      dataset_train_set = DATASET.head(dataset_sample_size)
      dataset_test_set = DATASET.tail(dataset_test_size)

    pca_reduction = set_param.pca_reduction(raw_pca_reduction, datset_feature_count)
    adaboost_estimators = set_param.adaboost_estimators(raw_adaboost_estimators)

  # Ensure all initializers have been set and are not None
  set_none_check = (
    DATASET_FILEPATH is None
    or datset_feature_count is None
    or datset_feature_start_col is None
    or datset_feature_end_col is None
    or dataset_sample_size is None
    or dataset_test_size is None
    or pca_reduction is None
    or adaboost_estimators is None
  )
  if set_none_check:
    print("Exiting. (ERR_SET_PARAM)")
    sys.exit(-1)

  print("\n=== Initialize Details ===")
  print("> Dataset: %s\n"
          "\t- Malignant Label: %s\n"
          "\t- Benign Label: %s\n"
          "\t- Feature Columns: [%s] - [%s]  (%s Features)\n"
          "\t- Sample Size: %s\n"
          "\t- Test Size: %s\n\n"
        "> PCA:\n"
          "\t- Reduction Size: %s\n\n"
        "> AdaBoost:\n"
          "\t- Estimators: %s\n"
        % (DATASET_FILEPATH.rsplit('/', 1)[-1], constants.MALIGNANT_LABEL, constants.BENIGN_LABEL,
          datset_feature_start_col, datset_feature_end_col,datset_feature_count,
          dataset_sample_size, dataset_test_size,
          pca_reduction,
          adaboost_estimators))

  if pca_reduction != 'none':
    reduction_values = pca.application.run (
                        pca_reduction,
                        dataset_train_set,
                        dataset_sample_size,
                        datset_feature_count,
                        datset_feature_start_col,
                        datset_feature_end_col
                      )
    if reduction_values is None:
      print("Exiting. (ERR_MIN_THRESHOLD)")
      sys.exit(-1)

  sys.exit(0)
