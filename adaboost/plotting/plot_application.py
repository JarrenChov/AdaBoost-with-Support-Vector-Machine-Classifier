import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from adaboost.plotting import methods

def run(estimators, train_error, test_error, train_accuracy, test_accuracy):
  print("Plotting...")

  fig, ax = plt.subplots(1,2, figsize = (12, 8))

#################
# Error Rate Plot
#################

  # Set X and Y axis range.
  ax[0].set_xlim(0, methods.set_max_xlim(estimators))
  ax[0].set_ylim(0, 0.6)

  # Set major and minor tick interval
  ax[0].xaxis.set_major_locator(MultipleLocator(methods.set_major_xaxis(estimators)))
  ax[0].xaxis.set_minor_locator(AutoMinorLocator(methods.set_minor_xaxis(estimators)))
  ax[0].yaxis.set_major_locator(MultipleLocator(0.1))
  ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))

  # Set grid major and minor tick interval style
  ax[0].grid(which='major', color='gainsboro', linestyle='--')
  ax[0].grid(which='minor', color='gainsboro', linestyle=':')

  # Plot data
  ax[0].plot([*range(estimators + 1)], test_error, label="test_error", color="lightcoral")
  ax[0].plot([*range(estimators + 1)], train_error, label="train_error", color="lightsteelblue")

  # Set display styling
  ax[0].set_title("Error Rate against Number of Estimators")
  ax[0].set_xlabel("Number of Estimators (Iterations)")
  ax[0].set_ylabel("Error Rate (%)")

  ax[0].legend(loc="upper right")


###############
# Accuracy Plot
###############

  # Set X and Y axis range.
  ax[1].set_xlim(0, methods.set_max_xlim(estimators))
  ax[1].set_ylim(0.4, 1.0)

  # Set major and minor tick interval
  ax[1].xaxis.set_major_locator(MultipleLocator(methods.set_major_xaxis(estimators)))
  ax[1].xaxis.set_minor_locator(AutoMinorLocator(methods.set_minor_xaxis(estimators)))
  ax[1].yaxis.set_major_locator(MultipleLocator(0.1))
  ax[1].yaxis.set_minor_locator(AutoMinorLocator(5))

  # Set grid major and minor tick interval style
  ax[1].grid(which='major', color='gainsboro', linestyle='--')
  ax[1].grid(which='minor', color='gainsboro', linestyle=':')

  # Plot data
  ax[1].plot([*range(estimators + 1)], train_accuracy, label="train_error", color="lightsteelblue")
  ax[1].plot([*range(estimators + 1)], test_accuracy, label="test_error", color="lightcoral")

  # Set display styling
  ax[1].set_title("Accuracy against Number of Estimators")
  ax[1].set_xlabel("Number of Estimators (Iterations)")
  ax[1].set_ylabel("Accuracy (%)")

  ax[1].legend(loc="lower right")

  plt.show()

  return 0
