import pandas as pd
from adaboost.common import constants

def dataset_features(dataset, feature_start_col, feature_end_col):
	return dataset.iloc[:, feature_start_col : feature_end_col]

def dataset_labels(dataset_filepath):
  dataset = pd.read_csv(dataset_filepath, header=None)
  dataset.iloc[:,1].replace(['M','B'],[constants.MALIGNANT_LABEL, constants.BENIGN_LABEL], inplace=True)
  return dataset

def reduce_dimension():
  return 0
