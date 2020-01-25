import numpy as np
from adaboost.common import format_dataset

def svm_prediction(dataset, w, b):
  return np.sign(np.dot(dataset, w) + b)

def prediction_comparison(prediction, dataset_label):
  return np.hstack((prediction == dataset_label))

def prediction_accuracy(prediction, dataset_label):
  shaped_prediction = format_dataset.reshape_vertical_float(prediction)
  shaped_label = format_dataset.reshape_vertical_float(dataset_label)

  comparison = prediction_comparison( shaped_prediction, shaped_label)
  predicted = len(np.where(comparison == True)[0])
  misclassified = len(np.where(comparison == False)[0])
  print(predicted)
  print(misclassified)
  print(predicted / (predicted + misclassified))
