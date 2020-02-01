from adaboost.common import constants

# Find difference from estimator to next nearest interval range
def interval_difference(interval, estimators):
  difference = interval - (estimators % interval)
  if difference != interval:
    return difference

  return 0


# Static major interval spacings for readable graph plotting
def lookup_interval_major(estimators):
  if estimators >= 150:
    return 50

  if estimators >= 80:
    return 20

  if estimators >= 40:
    return 10

  if estimators >= 20:
    return 5

  if estimators >= 10:
    return 2

  # Default major interval spacing
  return 1


# Static minor interval spacings for readable graph plotting
def lookup_interval_minor(interval):
  if interval >= 5:
    return 5

  # Default minor interval spacing
  return 4


# Set maximum x-limit of plot along x-axis
def set_max_xlim(estimators):
  xlim_max = estimators
  if xlim_max > 10:
    xlim_max = estimators + interval_difference(lookup_interval_major(estimators), estimators)

  if constants.OUTPUT_DETAIL is True:
    print("  --set Plot-XAxis-Max: %s" % (xlim_max))

  return xlim_max


# Set major interval spacing for along x-axis
def set_major_xaxis(estimators):
  interval = lookup_interval_major(estimators)
  if constants.OUTPUT_DETAIL is True:
    print("  --set Plot-XAxis-Interval-Major: %s" % (interval))

  return interval


# Set minor interval spacing for along x-axis
def set_minor_xaxis(estimators):
  major_interval = lookup_interval_major(estimators)
  minor_interval = lookup_interval_minor(major_interval)

  if constants.OUTPUT_DETAIL is True:
    print("  --set Plot-XAxis-Interval-Minor: %s" % (minor_interval))

  return minor_interval
