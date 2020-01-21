import numpy as np
from adaboost.common import constants
from adaboost.learning.weak_learner.classifier.svm import methods

def run(dataset, dataset_label, C):
  dim_samples, dim_features = dataset.shape
  shaped_label = dataset_label.values.reshape(-1, 1).astype(float)

  # Obtain quadratic programming problem parameters
  if constants.OUTPUT_DETAIL is True:
    print("--init SVM-Matrix-Construct")
  P, q, G, h, A, b = methods.dual_problem_quadratic_param (
                        dataset, shaped_label,
                        dim_samples,
                        C
                      )

  # Solve the quadratic programming problem
  if constants.OUTPUT_DETAIL is True:
    print("--init SVM-Quadratic-Programming-Problem-Solver")
  solution = methods.dual_problem_quadratic_solver(P, q, G, h, A, b)

  # Calculate (lagrange multiplier) alpha values
  if constants.OUTPUT_DETAIL is True:
    print("--init SVM-Lagrange-Multipliers")
  alphas = methods.svm_lagrange_multipliers(solution)

  # Find SVM support vectors that have a non-zero lagrange multiplier
  if constants.OUTPUT_DETAIL is True:
    print("--init SVM-Support-Vectors")
  S = methods.svm_support_vectors(alphas)

  # Find
  if constants.OUTPUT_DETAIL is True:
    print("--init SVM-Weight")
  w = methods.svm_weights(dataset, shaped_label, alphas)

  # Calculate the bias term
  if constants.OUTPUT_DETAIL is True:
    print("--init SVM-Bias")
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

  print("\n>> Initialized SVM Parameter Details\n"
        # "\t Margin Width (Maximized): %s  (Counts)\n"
        "\t Support Vectors: %s  (Counts)\n"
        "\t w (Weights): %d  (Counts)\n"
        "\t b (Bias) Value: %s\n"
        % (len(alphas[S].flatten()), len(w.flatten()), b[0]))

  prediction = methods.svm_classification_prediction(dataset, w, b)
  accuracy = methods.svm_classification_prediction_accuracy(shaped_label, prediction)
  # if constants.OUTPUT_DETAIL is True:
  #   print(">> SVM Classification Details")
  # print(np.where(accuracy)[0])


