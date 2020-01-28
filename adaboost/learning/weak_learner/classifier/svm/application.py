import numpy as np
from adaboost.common import constants, format_dataset
from adaboost.common.check import check_application
from adaboost.learning.weak_learner.classifier.svm import methods

# ===============
# SVM application
# ===============

def run(dataset, dataset_label, C, distribution_weights):
  dim_samples, dim_features = dataset.shape
  shaped_label = format_dataset.reshape_vertical_float(dataset_label)

  # Obtain quadratic programming problem parameters
  P, q, G, h, A, b = methods.dual_problem_quadratic_param (
                        dataset, shaped_label,
                        dim_samples,
                        C,
                        distribution_weights
                      )

  # Solve the quadratic programming problem
  solution = methods.dual_problem_quadratic_solver(P, q, G, h, A, b)

  # Calculate (lagrange multiplier) alpha values
  alphas = methods.svm_lagrange_multipliers(solution)

  # Find SVM support vectors that have a non-zero lagrange multiplier
  S = methods.svm_support_vectors(alphas)

  # Find weights
  w = methods.svm_weights(dataset, shaped_label, alphas)

  # Calculate the bias term
  b = methods.svm_bias(dataset, shaped_label, S, w)

  # Check neither of paramters are empty
  if check_application.zero_length_check([S, b]):
    if len(S) == 0:
      print("No support vectors found.")

    if len(b) == 0:
      print("Non existent bias value 'b'.")

    return None, None

  if constants.OUTPUT_DETAIL is True:
    print("")
  print("  SVM Parameter Details:\n"
        "\tMargin Width (Maximized): %s\n"
        "\tSupport Vectors: %s  (Counts)\n"
        "\tb (Bias) Value: %s"
        % ('%.4f' % methods.svm_max_margin(w),
          len(alphas[S].flatten()), b[0]))
  if constants.OUTPUT_DETAIL is True:
    print("")

  return w, b[0]
