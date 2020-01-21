import pandas as pd
from adaboost.common import constants

# Return specified column(s) from dataset
def dataset_extract_columns(dataset, columns):
  return dataset.iloc[:, columns]


# Return all dataset features based on starting and ending index
def dataset_extract_features(dataset, feature_start_col, feature_end_col):
	return dataset.iloc[:, feature_start_col : feature_end_col]


# Replace labels of default dataset from [M, B] to a binary classification [1, -1]
def dataset_default_label(dataset_filepath):
  dataset = pd.read_csv(dataset_filepath, header=None)
  dataset.iloc[:,1].replace(['M','B'],[constants.MALIGNANT_LABEL, constants.BENIGN_LABEL], inplace=True)
  return dataset


# Convert a np.ndarray to a pandas dataframe form
def pandas_dataframe(dataset):
  return pd.DataFrame(dataset)
