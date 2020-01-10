import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from adaboost.common import constants
from adaboost.common.get import retrieve_param
from adaboost.common.set import set_param

def run():
  DATASET = None

  DATASET_FILEPATH = None
  DATASET_SAMPLE_SIZE = None
  DATASET_TEST_SIZE = None
  ADABOOST_ESTIMATORS = None
  constants.initialize()

  raw_dataset = None
  raw_sample_size = None
  raw_estimators = None
  raw_out_detail = None

  # Extract parameter values from user input or application arguments
  if len(sys.argv) > 1:
    print("Retrieving parameters from arguments...")
    raw_dataset = retrieve_param.arg_dataset()
    raw_dataset_sample_size = retrieve_param.arg_dataset_sample_size()
    raw_adaboost_estimators = retrieve_param.adaboost_arg_estimators()
    raw_out_detail = retrieve_param.arg_out_detail()
  else:
    print("Retrieving parameters from user input...")
    raw_dataset = retrieve_param.input_dataset()
    raw_dataset_sample_size = retrieve_param.input_dataset_sample_size()
    raw_adaboost_estimators = retrieve_param.adaboost_input_estimators()
    raw_out_detail = retrieve_param.input_out_detail()

  # Ensure all parameters have a valid field
  retrieve_none_check = (
    raw_dataset is None
    or raw_dataset_sample_size is None
    or raw_adaboost_estimators is None
    or raw_out_detail is None
  )
  if retrieve_none_check:
    print("Exiting. (ERR_UNSET_PARAM)")
    exit(-1)

  # Initialize application variables with extracted parameter values
  if constants.OUTPUT_DETAIL is True:
    print("Initialize:")

  set_param.out_detail(raw_out_detail)

  DATASET_FILEPATH = set_param.dataset(raw_dataset)
  if DATASET_FILEPATH is not None:
    DATASET = pd.read_csv(DATASET_FILEPATH, header=None)

    if raw_dataset == 'default':
      if constants.OUTPUT_DETAIL is True:
        print("--set DataSet-Label")
      DATASET.iloc[:,1].replace(['M','B'],[constants.MALIGNANT_LABEL,constants.BENIGN_LABEL], inplace=True)
    DATASET_SAMPLE_SIZE, DATASET_TEST_SIZE = set_param.dataset_sample_test_size(raw_dataset_sample_size, DATASET.shape[0])

    ADABOOST_ESTIMATORS = set_param.adaboost_estimators(raw_adaboost_estimators)

  # Ensure all initializers have been set and are not None
  set_none_check = (
    DATASET_FILEPATH is None
    or DATASET_SAMPLE_SIZE is None
    or DATASET_TEST_SIZE is None
    or ADABOOST_ESTIMATORS is None
  )
  if set_none_check:
    print("Exiting. (ERR_SET_PARAM)")
    exit(-1)

  print("\nInitialize Details:")
  print("> Dataset: %s\n"
          "\t- Malignant Label: %s\n"
          "\t- Benign Label: %s\n"
          "\t- Sample Size: %s\n"
          "\t- Test Size: %s\n\n"
        "> AdaBoost:\n"
          "\t- Estimators: %s\n"
        % (DATASET_FILEPATH.rsplit('/', 1)[-1], constants.MALIGNANT_LABEL, constants.BENIGN_LABEL,
         DATASET_SAMPLE_SIZE, DATASET_TEST_SIZE,
         ADABOOST_ESTIMATORS))

  return 0
