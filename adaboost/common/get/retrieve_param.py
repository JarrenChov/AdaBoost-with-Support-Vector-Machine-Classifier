import sys
from adaboost.common import constants, convert_type
from adaboost.common.check import check_type
from adaboost.common.get import extract_value

####################
# Dataset Parameters
####################
# Extract filepath value from specified argument
def arg_dataset():
  result = [param for param in sys.argv if 'dataset_file' in param]

  if len(result) == 0:
    print("> Unspecified parameter found (Specify dataset: dataset_file={default | filepath}).")
  elif len(result) >= 2:
    print("> Multiple dataset found. Check only a single dataset is specified.\n"
          "\tFound argument 'dataset' duplicates: %s\n"
          % (result))
  else:
    return extract_value.arg_value('dataset_file', result[0])
  return None


# Extract filepath value from user specified value
def input_dataset():
  supplied_datasets = [1]
  accept_param_value = False
  result = None

  print("Please enter a dataset file path or select one from below:\n"
        "Existing Dataset files:\n"
        "\t( 1 ) Wisconsin Diagnostic Breast Cancer (WDBC) Dataset")

  while accept_param_value is False:
    result = input("Dataset Path / Number Selection: ").strip()
    param_value = convert_type.to_int(result)

    if param_value is None or result == '':
      param_value = convert_type.to_float(result)
      if check_type.is_float(param_value) or result == '':
        print("Check input is of <type 'int'> | <type 'str'> . Try Again.\n")
      else:
        print("Using supplied filepath: '%s'\n" % (result))
        accept_param_value = True
    elif check_type.is_int(param_value):
      if param_value > 0:
        if param_value in supplied_datasets:
          result = "default"
          accept_param_value = True
        else:
          print("Supplied dataset not found. Try Again.\n")
      else:
        print("Invalid dataset number. Try Again.\n")
  return result


# Extract dataset feature range and columns from specified argument
def arg_dataset_features_col():
  result = [param for param in sys.argv if 'dataset_feature_columns' in param]

  if len(result) == 0:
    print("> Unspecified parameter found (Specify dataset feature column range: dataset_feature_columns=<type 'int'>-<type 'int'>).")
  elif len(result) >= 2:
    print("> Multiple dataset feature columns found. Check only a single dataset feature column is specified.\n"
          "\tFound argument 'dataset feature column' duplicates: %s\n"
          % (result))
  else:
    if '-' in result[0]:
      return extract_value.arg_value('dataset_feature_columns', result[0])
  return None


# Extract dataset feature range and columns from user specified value
def input_dataset_features_col():
  accept_param_value = False
  result = None

  while accept_param_value is False:
    result = input("Dataset feature Column(s) range: ").strip().lower().replace(" ", '')

    if '-' in result:
      seperator = result.split("-", 1)
      start_col = seperator[0]
      end_col = seperator[1]

      start_col = convert_type.to_int(start_col)
      end_col = convert_type.to_int(end_col)

      if check_type.is_int(start_col) and check_type.is_int(end_col):
        if start_col >= end_col:
          print("Ensure feature range order is correct. Try Again.\n")
        else:
          if start_col >= 0 and end_col >= 0:
            accept_param_value = True
          else:
            print("Invalid number. Try Again.\n")
      else:
        print("Check inputs are of <type 'int'> . Try Again.\n")
    else:
      print("Ensure column numbers are seperated by a '-'. Try Again.\n")
    param_value = convert_type.to_int(result)
  return result


# Extract dataset label column from specified argument
def arg_dataset_label_col():
  result = [param for param in sys.argv if 'dataset_label_column' in param]

  if len(result) == 0:
    print("> Unspecified parameter found (Specify dataset label column: dataset_label_column=<type 'int'>).")
  elif len(result) >= 2:
    print("> Multiple dataset label columns found. Check only a single dataset label column is specified.\n"
          "\tFound argument 'dataset label column' duplicates: %s\n"
          % (result))
  else:
    return extract_value.arg_value('dataset_label_column', result[0])
  return None


# Extract dataset label column from user specified value
def input_dataset_label_col():
  accept_param_value = False
  result = None

  while accept_param_value is False:
    result = input("Dataset label Column: ").strip().lower()
    param_value = convert_type.to_int(result)

    if check_type.is_int(param_value):
      if param_value >= 0:
        accept_param_value = True
      else:
        print("Invalid number. Try Again.\n")
    else:
      print("Check input is of <type 'int'> . Try Again.\n")
  return result


# Extract training sample size value from specified argument
def arg_dataset_sample_size():
  result = [param for param in sys.argv if 'dataset_sample_size' in param]

  if len(result) == 0:
    print("> Unspecified parameter found (Specify dataset sample size: dataset_sample_size=<type 'int'>).")
  elif len(result) >= 2:
    print("> Multiple dataset sample sizes found. Check only a single sample size is specified.\n"
          "\tFound argument 'dataset sample size' duplicates: %s\n"
          % (result))
  else:
    return extract_value.arg_value('dataset_sample_size', result[0])
  return None


# Extract sample size value from user specified value
def input_dataset_sample_size():
  accept_param_value = False
  result = None

  while accept_param_value is False:
    result = input("Dataset Sample Size: ").strip()
    param_value = convert_type.to_int(result)

    if check_type.is_int(param_value):
      if param_value > 0:
        accept_param_value = True
      else:
        print("Invalid number. Try Again.\n")
    else:
      print("Check input is of <type 'int'> . Try Again.\n")
  return result


################
# PCA Parameters
################
# [OPTIONAL] Extract pca reduction value from specified argument
def arg_pca_reduction():
  result = [param for param in sys.argv if 'pca_reduction' in param]

  if len(result) == 0:
    print("> [OPTIONAL] (Specify pca reduction size: pca_reduction={\"default\" | <type 'int'> | <type 'float'>}).")
    print("\tUsing Default: PCA-Reduction=%s\n" % ('none'))
    return "none"
  elif len(result) >= 2:
    print("> Multiple pca reduction found. Check only a single pca reduction is specified.\n"
          "\tFound argument 'pca reduction' duplicates: %s\n"
          % (result))
  else:
    return extract_value.arg_value('pca_reduction', result[0])
  return None


# Extract pca reduction value from user specified value
def input_pca_reduction():
  accept_param_value = False
  result = None

  while accept_param_value is False:
    result = input("PCA reduction size on dataset (proportion [0 <-> 1] / Number of features): ").strip().lower()

    if result == "none" or result == "default":
      accept_param_value = True

    if accept_param_value is False:
      if '.' in result:
        param_value = convert_type.to_float(result)
        if check_type.is_float(param_value):
          if param_value >= 0 and param_value <= 1:
            accept_param_value = True
          else:
            print("Invalid number. Try Again.\n")
        else:
          print("Please check input is one of the following:\n"
          "\"default\" | \"none\" | <type 'int'> | <type 'float'>\n"
          "Try Again.\n")
      else:
        param_value = convert_type.to_int(result)
        if check_type.is_int(param_value):
          if param_value >= 0:
            accept_param_value = True
          else:
            print("Invalid number. Try Again.\n")
        else:
          print("Please check input is one of the following:\n"
          "\"default\" | \"none\" | <type 'int'> | <type 'float'>\n"
          "Try Again.\n")
  return result


################
# SVM Parameters
################
def arg_svm_regularizer_c():
  result = [param for param in sys.argv if 'svm_regularizer_c' in param]

  if len(result) == 0:
    print("> [OPTIONAL] (Specify pca reduction size: svm_regularizer_c={\"default\" | \"none\" | <type 'float'>}).")
    print("\tUsing Default: SVM-Regularizer-C=%s\n" % ('1.0'))
    return "default"
  elif len(result) >= 2:
    print("> Multiple SVM regularizer C found. Check only a single SVM regularizer C is specified.\n"
          "\tFound argument 'SVM regularizer C' duplicates: %s\n"
          % (result))
  else:
    return extract_value.arg_value('svm_regularizer_c', result[0])
  return None


def input_svm_regularizer_c():
  accept_param_value = False
  result = None

  while accept_param_value is False:
    result = input("SVM regularizer parameter C: ").strip().lower()

    if result == "none" or result == "default":
      accept_param_value = True

    if accept_param_value is False:
      param_value = convert_type.to_float(result)

      if check_type.is_float(param_value):
        if param_value >= 0.0:
          if param_value == 0.0:
            result = 'none'

          if param_value == 1.0:
            result = 'default'
          accept_param_value = True
        else:
          print("Invalid number. Try Again.\n")
      else:
        print("Check input is one of the following:\n"
        "\"default\" | \"none\" | <type 'float'>\n"
        "Try Again.\n")
  return result

#####################
# AdaBoost Parameters
#####################
# Extract estimator value from specified argument
def arg_adaboost_estimators():
  result = [param for param in sys.argv if 'adaboost_estimators' in param]

  if len(result) == 0:
    print("> Unspecified parameter found (Specify adaboost estimator: adaboost_estimators=<type 'int'>).")
  elif len(result) >= 2:
    print("> Multiple adaboost estimators found. Check only a single adaboost estimator is specified.\n"
          "\tFound argument 'adaboost estimator' duplicates: %s\n"
          % (result))
  else:
    return extract_value.arg_value('adaboost_estimators', result[0])
  return None


# Extract estimator value from user specified value
def input_adaboost_estimators():
  accept_param_value = False
  result = None

  while accept_param_value is False:
    result = input("Number of AdaBoost Estimators (Weak Learners): ").strip()
    param_value = convert_type.to_int(result)

    if check_type.is_int(param_value):
      if param_value > 0:
        accept_param_value = True
      else:
        print("Invalid number. Try Again.\n")
    else:
      print("Check input is of <type 'int'> . Try Again.\n")
  return result


#######################
# Out Detail Parameters
#######################
# [OPTIONAL] Extract output detail value from specified argument
def arg_out_detail():
  result = [param for param in sys.argv if 'output_detail' in param]

  if len(result) == 0:
    print("> [OPTIONAL] (Specify output detail: output_detail={true | false}).")
    print("\tUsing Default: Out-Detail=%s\n" %(constants.OUTPUT_DETAIL))
  elif len(result) >= 2:
    print("> Multiple output details found. Check only a single output detail is specified.\n"
          "\tFound argument 'output detail' duplicates: %s\n"
          % (result))
    return None
  else:
    return extract_value.arg_value('output_detail', result[0])
  return constants.OUTPUT_DETAIL

# Extract output detail value from user specified value
def input_out_detail():
  accepted_params = [ 'yes', 'y', 'no', 'n' ]
  accept_param_value = False
  result = None

  while accept_param_value is False:
    result = input("Output detailed information? (yes/no): ").strip().lower()

    if result in accepted_params:
      if 'y' in result:
        result = 'true'
      elif 'n' in result:
        result = 'false'
      accept_param_value = True
    else:
      print("Please enter one of the following:\n"
      "%s\n"
      "Try Again.\n"
      % (accepted_params))
  print()
  return result
