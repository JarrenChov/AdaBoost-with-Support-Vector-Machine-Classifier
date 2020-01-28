import numpy as np
from adaboost.common import classes, format_dataset

# Predict output labels for a SVM model
def svm_prediction(dataset, w, b):
  return np.sign(np.dot(dataset, w) + b)


# Compare only labels between predicted and actual,
# Returns a list
def prediction_comparison(prediction, dataset_label):
  return np.hstack((prediction == dataset_label))


# Calculate prediction accuracy between predicted and actual
def prediction_accuracy(prediction, dataset_label):
  result = classes.classify()

  shaped_prediction = format_dataset.reshape_vertical_float(prediction)
  shaped_label = format_dataset.reshape_vertical_float(dataset_label)

  # Obtain comparison match between predicted and real labels
  comparison = prediction_comparison(shaped_prediction, shaped_label)

  # Obtain number of correctly and wrongly classified labels
  result.classified = len(np.where(comparison == True)[0])
  result.misclassified = len(np.where(comparison == False)[0])

  result.classified_accuracy = result.classified / (result.classified + result.misclassified)
  result.classified_error = 1 - result.classified_accuracy

  result.misclassified_accuracy = result.misclassified / (result.classified + result.misclassified)
  result.misclassified_error = 1 - result.misclassified_accuracy

  return result
