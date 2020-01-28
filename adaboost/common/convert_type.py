from adaboost.common import constants
from adaboost.common.check import check_type

# Convert a string type true / false to a boolean type
def to_bool(data):
  # Check type is already a boolean
  if check_type.is_bool(data):
    return data

  # Check type is not a empty, float, integer, list
  type_check = (
    check_type.is_float(data)
    or check_type.is_int(data)
    or check_type.is_list(data)
    or data is None
    or data == ''
  )

  if type_check:
    if data is None or data == '':
      if constants.OUTPUT_DETAIL is True:
        print("Invalid type: Empty String")
    else:
      if constants.OUTPUT_DETAIL is True:
        print("Invalid type: ", data, type(data))
    return None

  bool_value = data.lower().capitalize()
  if bool_value == 'True':
    return True
  elif bool_value == 'False':
    return False
  else:
    if constants.OUTPUT_DETAIL is True:
      print("Failed to convert '%s' to boolean" % (data))

  return None


# Convert a string or integer type to a float type
def to_float(data):
  float_value = None

  # Check type is a boolean, empty, list
  type_check = (
    check_type.is_list(data)
    or check_type.is_bool(data)
    or data is None
    or data == ''
  )

  if type_check:
    if data is None or data == '':
      if constants.OUTPUT_DETAIL is True:
        print("Invalid type: Empty String")
    else:
      if constants.OUTPUT_DETAIL is True:
        print("Invalid type: ", data, type(data))
    return None

  # Check type is already a float
  if check_type.is_float(data):
    return data

  try:
    float_value = float(data)
  except ValueError:
    if constants.OUTPUT_DETAIL is True:
      print("Failed to convert '%s' to Float" % (data))

  return float_value


# Convert a string type to a integer type
def to_int(data):
  int_value = None

  # Check type is a boolean, empty, float, list
  type_check = (
    check_type.is_list(data)
    or check_type.is_float(data)
    or check_type.is_bool(data)
    or data is None
    or data == ''
  )

  if type_check:
    if data is None or data == '':
      if constants.OUTPUT_DETAIL is True:
        print("Invalid type: Empty String")
    else:
      if constants.OUTPUT_DETAIL is True:
        print("Invalid type: ", data, type(data))
    return None

  # Check type is already a integer
  if check_type.is_int(data):
    return data

  try:
    int_value = int(data)
  except ValueError:
    if constants.OUTPUT_DETAIL is True:
      print("Failed to convert '%s' to Integer" % (data))

  return int_value


# Convert integer, float, boolean types to a string type
def to_string(data):
  # Check type is a list or empty
  type_check = (
    check_type.is_list(data)
    or data is None
    or data == ''
  )

  if type_check:
    if data is None or data == '':
      if constants.OUTPUT_DETAIL is True:
        print("Invalid type: Empty String")
    else:
      if constants.OUTPUT_DETAIL is True:
        print("Invalid type: ", data, type(data))
    return None

  # Check type is already a string
  if check_type.is_str(data):
    return data

  return str(data)
