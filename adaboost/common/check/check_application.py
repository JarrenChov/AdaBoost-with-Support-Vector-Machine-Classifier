import math
import sys

# Check if application should print out help
def application_help_check():
  helper = [param for param in sys.argv if 'help' in param]

  if len(helper) == 1:
    if helper[0] == "help":
      return True

  return False


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


# Check if application is run for graphical plotting performance
def plot_application_check():
  plot = [param for param in sys.argv if 'plot' in param]

  if len(plot) == 1:
    if plot[0] == "plot":
      return True

  return False


# Check if values are 0
def zero_length_check(params):
  for param in params:
    if len(param) == 0:
      return True

  return False
