import cvxopt
import numpy as np
from adaboost.common import constants

# Construct matrix's for P, q, G, h, A, b as a CVXOPT format
def dual_problem_quadratic_param(dataset, dataset_label, dim_samples, C, distribution_weights):
  if constants.OUTPUT_DETAIL is True:
    print("  --init SVM-Matrix-Construct")

  x_prime = dataset_label * dataset.values
  gram_matrix = np.dot(x_prime, x_prime.T).astype(float)

  P = cvxopt.matrix(gram_matrix)
  q = cvxopt.matrix(np.ones((dim_samples, 1)) * -1)

  # Defaults to soft margin SVM,
  # However, if C is 0 Hard margin SVM is used
  if C == 0.0:
    if constants.OUTPUT_DETAIL is True:
      print("  --set -data=linearly-separable -type=hard-margin")

    G = cvxopt.matrix(np.eye(dim_samples) * -1)
    h = cvxopt.matrix(np.zeros(dim_samples))
  else:
    if constants.OUTPUT_DETAIL is True:
      print("  --set -data=non-linearly-separable -type=soft-margin")

    G = cvxopt.matrix(np.vstack(((np.eye(dim_samples) * -1), np.eye(dim_samples))))

    # Calc h using default C if no distribution weights given
    if distribution_weights is None:
      h = cvxopt.matrix(np.hstack((np.zeros(dim_samples), np.ones(dim_samples) * C)))
    else:
      shaped_distribution_weight = distribution_weights.values.flatten().astype(float)
      h = cvxopt.matrix(np.hstack((np.zeros(dim_samples), shaped_distribution_weight * (np.ones(dim_samples) * C))))

  A = cvxopt.matrix(dataset_label.reshape(1, -1))
  b = cvxopt.matrix(np.zeros(1))

  return P, q, G, h, A, b


# Solve the problem as a quadratic
def dual_problem_quadratic_solver(P, q, G, h, A, b):
  if constants.OUTPUT_DETAIL is True:
    print("  --init SVM-Quadratic-Programming-Problem-Solver")

    cvxopt.solvers.options['show_progress'] = True
  else:
    cvxopt.solvers.options['show_progress'] = False

  return cvxopt.solvers.qp(P, q, G, h, A, b)


# Obtain alpha values from lagrange multipliers as a list
def svm_lagrange_multipliers(solution):
  if constants.OUTPUT_DETAIL is True:
    print("\n  --init SVM-Lagrange-Multipliers")

  return np.array(solution['x'])


# Obtain support vectors from lagrange multipliers which are greather then the threshold (non-zero)
def svm_support_vectors(alpha):
  if constants.OUTPUT_DETAIL is True:
    print("  --init SVM-Support-Vectors")

  return (alpha > constants.NON_ZERO_LAGRANGE_THRESHOLD).flatten()


# Obtain weighted vector
def svm_weights(dataset, dataset_label, alpha):
  if constants.OUTPUT_DETAIL is True:
    print("  --init SVM-Weight")

  return np.dot((dataset_label * alpha).T, dataset).reshape(-1, 1)


# Obtain bias value
def svm_bias(dataset, dataset_label, S, w):
  if constants.OUTPUT_DETAIL is True:
    print("  --init SVM-Bias")

  return dataset_label[S] - np.dot(dataset[S], w)


# Obtain max marign of SVM boundary
def svm_max_margin(w):
  return 2 / np.linalg.norm(w)
