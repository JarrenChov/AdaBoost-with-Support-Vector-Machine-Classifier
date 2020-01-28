import math

# Check if any of the default datasets have been used
def default_datasets(value):
  if value == 'default_1':
    return True

  if value == 'default_2':
    return True

  if value == 'default_3':
    return True

  return False


# Check supplied dataset uses required label foramt {-1, +1}
def label_value_check(labels):
  # Define absolute tolerance value
  tol = 0.0

  if (
    math.isclose(labels[0], 1.0, abs_tol = tol)
    and math.isclose(labels[1], -1.0, abs_tol = tol)
  ):
    return True

  if (
    math.isclose(labels[0], -1.0, abs_tol = tol)
    and math.isclose(labels[1], 1.0, abs_tol = tol)
  ):
    return True

  return False


# Check if values have not been initalized
def none_check(params):
  for param in params:
    if param is None:
      return True

  return False


# Check if values are 0
def zero_length_check(params):
  for param in params:
    if len(param) == 0:
      return True

  return False
