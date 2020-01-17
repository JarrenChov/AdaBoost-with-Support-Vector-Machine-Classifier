from bisect import bisect
import numpy as np
import pandas as pd
from adaboost.common import convert_type
from adaboost.common.check import check_type

# Calculate per feature (column) mean value
# Return vertical feature list as a horizontal list
def calculate_mean(dataset_features):
  raw_feature_mean = dataset_features.mean().values
  return raw_feature_mean.transpose()


# Return a normalized dataset
def normalize_dataset(dataset_features, feature_mean, sample_size):
  mean_matrix = np.repeat(feature_mean[None, :], sample_size, axis=0)
  return dataset_features.values - mean_matrix


# Calculate covariance matrix
def calculate_covariance(normalized_features, feature_count):
  return (1 / (feature_count - 1)) * np.dot(normalized_features.T, normalized_features)


# Calculate eigenvalues and eigenvectors, along with its sorted eigenvalues.
def calculate_eigen_decomposition(covariance_matrix):
  eigenvalue, eigenvector = np.linalg.eig(covariance_matrix)
  # print(eigenvalue.real)
  sorted_eigen_index = np.argsort(-eigenvalue.real)
  return eigenvalue, eigenvector, sorted_eigen_index


# Reduce feature dataset dimensiosn based on either:
#   - A cumlative variance percentage of the total dataset up to a user specified variance threshold
#   - Default space by taking first x ammount of features until a variance of 1 is reached
#   - A user specified feature range
def reduce_dimensionality(reduction_size, sorted_eigen_index, eigenvalue):
  reduced_feature_size = None
  reduced_projection = None
  reduced_eigen_index = None

  if check_type.is_float(reduction_size) or check_type.is_str(reduction_size):
    cumlative_variance = np.cumsum(eigenvalue[sorted_eigen_index].real / np.sum(eigenvalue.real))
    # print(sorted_eigen_index.real)

    # Reduce dimension based on a cumlative variance totaling a variance threshold
    if check_type.is_float(reduction_size):
      feature_range = (bisect(cumlative_variance, reduction_size)) + 1
      if feature_range <= 1:
        print("Feature minimum threshold unreached. (min=2)\n"
              "Please try again within reduction range [%s <-> 1]."
              % (cumlative_variance[0]))
        return None
      else:
        reduced_feature_size = feature_range
    # Reduce dimensions based on the first instance of variance totaling 1.00000000
    elif reduction_size == "default":
      cumlative_variance = ['%.8f' % feature for feature in cumlative_variance]
      reduced_feature_size = cumlative_variance.index('1.00000000')
  # Reduce dimensions based on a specified feature range
  elif check_type.is_int(reduction_size):
    reduced_feature_size = reduction_size

  reduced_eigen_index = sorted_eigen_index[:reduced_feature_size]
  return [reduced_feature_size, reduced_eigen_index]


# Project sorted eigenvectors onto initial feature dataset
def data_projection(normalized_features, eigenvector, sorted_eigen_index):
  return np.dot(normalized_features, eigenvector[:, sorted_eigen_index].real)
