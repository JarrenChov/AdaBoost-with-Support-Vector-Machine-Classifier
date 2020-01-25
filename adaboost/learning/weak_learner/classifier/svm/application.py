import numpy as np
from adaboost.common import constants, format_dataset
from adaboost.learning.weak_learner.classifier.svm import methods

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

  len_zero_check = (
    len(S) == 0
    or len(b) == 0
  )
  if len_zero_check:
    if len(S) == 0:
      print("No support vectors found.")

    if len(b) == 0:
      print("Non existent bias value 'b'.")

    print("Exiting.")
    return None

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
