import os.path
from adaboost.common import constants, convert_type
from adaboost.common.check import check_type

# Before assinging, check dataset is exists as a file, or
# convert a supplied dataset to its default file path
def dataset(value):
  dataset_path = None
  dataset_filename = None

  if value == 'default':
    dataset_path = constants.WDBC_DATASET_PATH
    dataset_filename = constants.WDBC_DATASET_PATH.rsplit('/', 1)[-1]
  else:
    if not check_type.is_str(value):
      print("Invalid dataset file (File Not Found). Exiting.")
      return None

    if os.path.isfile(value):
      dataset_path = value
      dataset_filename = value.rsplit('/', 1)[-1]
    else:
      print("Invalid dataset file (File Not Found). Exiting.")
      return None

  if constants.OUTPUT_DETAIL is True:
    print("--set Dataset-File: %s" % (dataset_filename))
  return dataset_path


# Before assinging, check data sample and test size are valid values
def dataset_sample_test_size(value, sample_size_max):
  # Check if value is of type integer or a string integer
  sample_size = convert_type.to_int(value)
  sample_size_max = convert_type.to_int(sample_size_max)

  value_check = (
    sample_size is None
    or sample_size <= 0
    or sample_size_max is None
    or sample_size_max <= 0
  )
  if value_check:
    print("Invalid dataset sample size (Empty Sample Size). Exiting.")
    return None, None

  test_size = sample_size_max - sample_size
  if test_size <= 0:
    print("Invalid dataset sample size (Sample Size Exceeds Dataset). Exiting.")
    return None, None

  if constants.OUTPUT_DETAIL is True:
    print("--set Dataset-Sample-Size: %d" % (sample_size))
    print("--set Dataset-Test-Size: %d" % (test_size))
  return sample_size, test_size


# Before assinging, check pca reduction ia a valid value
def pca_reduction(value, feature_count_max):
  reduction = None

  if value == "none":
    reduction = "none"
  elif value == "default":
    reduction = "default"
  else:
    print("--init -type PCA-Reduction: %s" % ("<class 'int'>"))
    reduction = convert_type.to_int(value)

    if reduction is None:
      print("--init -type PCA-Reduction: %s" % ("<class 'float'>"))
      reduction = convert_type.to_float(value)
      if reduction is not None:
        if reduction >= 0.0 and reduction <= 1.0:
          if reduction == 0.0:
            reduction = "none"

          if reduction == 1.0:
            reduction = "default"
        else:
          print("Invalid PCA reduction size (Reduction Size Exceeds Range [0 - 1]). Exiting.")
          return None
      else:
        print("Invalid PCA reduction size (Invalid Reduction Size). Exiting.")
        return None
    else:
      if reduction >= 0 and reduction <= feature_count_max:
        if reduction == 0:
          reduction = "none"

        if reduction == 1 or reduction == feature_count_max:
          reduction = "default"
      else:
        print("Invalid PCA reduction size (Reduction Size Exceeds Count [0 - %d]. Exiting." % (feature_count_max))
        return None

  if constants.OUTPUT_DETAIL is True:
    if check_type.is_int(reduction) or check_type.is_float(reduction):
      if check_type.is_float(reduction):
        print("--set PCA-Reduction: %s [Proportion]" % (reduction))
      else:
        print("--set PCA-Reduction: %d [Feature]" % (reduction))
    else:
      print("--set PCA-Reduction: %s" % (reduction))
  return reduction


# Before assinging, check adaboost estimator ia a valid value
def adaboost_estimators(value):
  estimators = convert_type.to_int(value)

  if estimators is None or estimators <= 0:
    print("Invalid estimator. Exiting.")
    return None

  if constants.OUTPUT_DETAIL is True:
    print("--set AdaBoost-Estimators: %d" % (estimators))
  return estimators


# Before assinging, check out detail is a valid value
def out_detail(value):
  out_detail_value = convert_type.to_bool(value)

  if out_detail_value is None:
    print("Invalid out detail value. Reverting to default.")
    return

  if constants.OUTPUT_DETAIL is True:
    print("--set Out-Detail: %s" % (out_detail_value))
  constants.OUTPUT_DETAIL = out_detail_value
  return
