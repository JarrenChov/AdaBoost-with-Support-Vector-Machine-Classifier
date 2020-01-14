from adaboost.common import format_dataset
from adaboost.learning.dimension_reduction.pca import methods

def run(reduction_size, dataset, sample_size, feature_count, feature_start_col, feature_end_col):
  reduced_feature_size = None
  reduced_projection_matrix = None
  reduced_eigen_index = None

  # Extract all features from dataframe
  dataset_features = format_dataset.dataset_features (
    dataset,
    feature_start_col,
    feature_end_col
  )

  # Calculate for each feature, its corresponding mean value
  feature_mean = methods.calculate_mean(dataset_features)

  # Normalize newly centered data
  normalized_features = methods.normalize_dataset (
    dataset_features,
    feature_mean,
    sample_size
  )

  # Create a covariance matrix
  covariance_matrix = methods.calculate_covariance (
    normalized_features,
    feature_count
  )

  # Find eigenvalues, eigenvectors, along with the sorted eigenvalue order
  eigenvalue, eigenvector, sorted_eigen_index = methods.calculate_eigen_decomposition(covariance_matrix)

  # Create a new feature matrix, by projecting eigenvectors onto the dataset features
  projection_matrix = methods.data_projection (
    normalized_features,
    eigenvector,
    sorted_eigen_index
  )

  # Reduce dimension of dataset features
  reduced_results = methods.reduce_dimensionality (
                      reduction_size,
                      projection_matrix,
                      sorted_eigen_index,
                      eigenvalue
                    )
  if reduced_results is None:
    return None

  reduced_feature_size = reduced_results[0]
  reduced_projection_matrix = reduced_results[1]
  reduced_eigen_index = reduced_results[2]
  return [reduced_feature_size, reduced_projection_matrix, reduced_eigen_index]
