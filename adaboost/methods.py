import numpy as np
from adaboost.common import constants, classification, format_dataset

def distribution_weights(sample_size):
  if constants.OUTPUT_DETAIL is True:
    print("  --init Adaboost-Distribution-Weight")

  d1_value = np.array([ [1/sample_size] ])
  return format_dataset.pandas_dataframe(np.repeat(d1_value, sample_size, axis=0))


def set_hypothesis_value(hypothesis, w, b):
  hypothesis.svm_w = w
  hypothesis.svm_b = b
  return hypothesis


def hypothesis_classification_error(hypothesis, dataset, dataset_label, distribution_weights):
  if constants.OUTPUT_DETAIL is True:
    print("  --init Adaboost-Hypothesis-Prediction")
    print("  --init Adaboost-Classification-Error")

  shaped_label = format_dataset.reshape_vertical_float(dataset_label)
  distribution_weights_list = distribution_weights.values.flatten()

  prediction = classification.svm_prediction(dataset, hypothesis.svm_w, hypothesis.svm_b)
  prediction_comparison = np.invert(np.hstack(classification.prediction_comparison(prediction, shaped_label)))
  misclassification = np.multiply(prediction_comparison, 1)

  return prediction, np.sum(distribution_weights_list * misclassification) / np.sum(distribution_weights_list)


def hypothesis_significance(error):
  if constants.OUTPUT_DETAIL is True:
    print("  --init Adaboost-significance-alpha")

  return 1/2 * np.log(((1 - error) / error) + constants.ZERO_SIGNIFICANCE_THRESHOLD)


def update_distribution_weights(alpha, prediction, dataset_label, distribution_weights):
  if constants.OUTPUT_DETAIL is True:
    print("  --init -update Adaboost-Distribution-Weight")

  shaped_label = format_dataset.reshape_vertical_float(dataset_label)

  updated_distribution = distribution_weights.values * np.exp(-1 * alpha * shaped_label * prediction)
  normalized_distribution = updated_distribution / np.sum(updated_distribution)

  return format_dataset.pandas_dataframe(normalized_distribution)


def hypothesis_final(model, dataset, dim_samples):
  continuos_hypothesis = np.ones(dim_samples).reshape(-1, 1).astype(float)

  for hypothesis in model:
    weak_learner = classification.svm_prediction(dataset, hypothesis.svm_w, hypothesis.svm_b)
    continuos_hypothesis = continuos_hypothesis + (hypothesis.model_weight *  weak_learner)

  return format_dataset.pandas_dataframe(np.sign(continuos_hypothesis))
