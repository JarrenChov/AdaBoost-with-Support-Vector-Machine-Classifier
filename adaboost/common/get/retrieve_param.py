import sys
from adaboost.common import constants, convert_type
from adaboost.common.check import check_type
from adaboost.common.get import extract_value

#####################
# AdaBoost Parameters
#####################
# Extract estimator value from specified argument
def adaboost_arg_estimators():
  result = [param for param in sys.argv if 'adaboost_estimators' in param]

  if len(result) == 0:
    print("> Unspecified parameter found (Specify estimator: estimators=<type 'int'>).")
  elif len(result) >= 2:
    print("> Multiple estimators found. Check only a single estimator is specified.\n"
          "\tFound argument 'estimator' duplicates: %s\n"
          % (result))
  else:
    return extract_value.arg_value('adaboost_estimators', result[0])
  return None


# Extract estimator value from user specified value
def adaboost_input_estimators():
  accept_param_value = False
  result = None

  while accept_param_value is False:
    result = input("Number of Estimators (Weak Learners): ").strip()
    param_value = convert_type.to_int(result)

    if check_type.is_int(param_value):
      if param_value > 0:
        accept_param_value = True
      else:
        print("Invalid number. Try Again.\n")
    else:
      print("Check input is of <type 'int'> . Try Again.\n")
  return result


####################
# Dataset Parameters
####################
# Extract filepath value from specified argument
def arg_dataset():
  result = [param for param in sys.argv if 'dataset_file' in param]

  if len(result) == 0:
    print("> Unspecified parameter found (Specify dataset: file={default | filepath}).")
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
    print("> Unspecified parameter found (Specify sample size: sample_size=<type 'int'>).")
  elif len(result) >= 2:
    print("> Multiple sample sizes found. Check only a single sample size is specified.\n"
          "\tFound argument 'sample size' duplicates: %s\n"
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
    result = input("Training Sample Size: ").strip()
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
