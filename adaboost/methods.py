import numpy as np
from adaboost.common import constants, classification, format_dataset

# Initialize as a pandas dataframe, distribution weights for each sample as 1/samples_size
def distribution_weights(sample_size):
  if constants.OUTPUT_DETAIL is True:
    print("  --init Adaboost-Distribution-Weight\n")

  d1_value = np.array([ [1/sample_size] ])
  return format_dataset.pandas_dataframe(np.repeat(d1_value, sample_size, axis=0))


# Assign corresponding svm w and b values to hypothesis model
def set_hypothesis_value(hypothesis, w, b):
  hypothesis.svm_w = w
  hypothesis.svm_b = b
  return hypothesis


# Calculate prediction error percentage by obtained hypothesis model
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


# Calculate hypothesis model significance (alpha) in final model
def hypothesis_significance(error):
  if constants.OUTPUT_DETAIL is True:
    print("  --init Adaboost-significance-alpha")

  return 1/2 * np.log(((1 - error) / error) + constants.ZERO_SIGNIFICANCE_THRESHOLD)


# Update distribution weights, by increasing importance on wrongly classified,
# and decreasing importance on correctly classified points
def update_distribution_weights(alpha, prediction, dataset_label, distribution_weights):
  if constants.OUTPUT_DETAIL is True:
    print("  --init -update Adaboost-Distribution-Weight")

  shaped_label = format_dataset.reshape_vertical_float(dataset_label)
  updated_distribution = distribution_weights.values * np.exp(-1 * alpha * shaped_label * prediction)
  normalized_distribution = updated_distribution / np.sum(updated_distribution)

  return format_dataset.pandas_dataframe(normalized_distribution)


# Generate final outcome by combing all hypothesis models into a single model
def hypothesis_final(model, dataset, dim_samples):
  continuos_hypothesis = np.zeros(dim_samples).reshape(-1, 1).astype(float)

  # Run first model as a SVM, without AdaBoost (Only used in application_plot)
  if len(model) == 1 and model[0].model_weight == -1:
    initial_svm = classification.svm_prediction(dataset, model[0].svm_w, model[0].svm_b)
    continuos_hypothesis = continuos_hypothesis + initial_svm
  else:
    for hypothesis in model:
      # Only process models which have a weight > 0
      if hypothesis.model_weight != -1:
        weak_learner = classification.svm_prediction(dataset, hypothesis.svm_w, hypothesis.svm_b)
        continuos_hypothesis = continuos_hypothesis + (hypothesis.model_weight * weak_learner)

  return format_dataset.pandas_dataframe(np.sign(continuos_hypothesis))
