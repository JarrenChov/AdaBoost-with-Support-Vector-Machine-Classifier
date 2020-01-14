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
def pca_arg_reduction():
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
def pca_input_reduction():
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
          if param_value > 0:
            accept_param_value = True
          else:
            print("Invalid number. Try Again.\n")
        else:
          print("Please check input is one of the following:\n"
          "\"default\" | \"none\" | <type 'int'> | <type 'float'>\n"
          "Try Again.\n")
  return result


#####################
# AdaBoost Parameters
#####################
# Extract estimator value from specified argument
def adaboost_arg_estimators():
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
def adaboost_input_estimators():
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
    print("> [OPTIONAL] (Specify sample size: output_detail={true | false}).")
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
