
import numpy as np
import pandas as pd
from adaboost.common import constants, convert_type
from adaboost.common.check import check_application, check_type

# Return specified column(s) from dataset
def dataset_extract_columns(dataset, columns):
  return dataset.iloc[:, columns]


# Return all dataset features based on starting and ending index
def dataset_extract_features(dataset, feature_start_col, feature_end_col):
	return dataset.iloc[:, feature_start_col : feature_end_col]


# Replace labels of dataset, to required labels format of [1, -1]
# If labels consists of [1, -1], corresponding labels will be used instead
def dataset_default_label(dataset_filepath, label_col):
  USE_SUPPLIED_LABEL = False

  dataset = pd.read_csv(dataset_filepath, header=None)
  dataset_labels = dataset.iloc[:, label_col].unique().tolist()
  default_label = [constants.DEFAULT_LABEL_0, constants.DEFAULT_LABEL_1]

  # Check problem is only a binary classification (no support for multi-class SVM)
  if len(dataset_labels) == 2:
    if type(dataset_labels[0]) != type(dataset_labels[1]):
      print("Inconsistent dataset label type. Found:[ %s, %s ] (Dataset Label Type Mismatch)."
            % (type(dataset_labels[0]), type(dataset_labels[1])))
      return None, None

    # Check if labels are of required labels {-1, 1}
    if check_type.is_int(dataset_labels[0]) or check_type.is_float(dataset_labels[0]):
      if check_application.label_value_check(dataset_labels):
        USE_SUPPLIED_LABEL = True

    if USE_SUPPLIED_LABEL is True:
      if constants.OUTPUT_DETAIL is True:
        print("  --set Dataset-Labels")

      default_label = ["Default Label", "Default Label"]
    else:
      if constants.OUTPUT_DETAIL is True:
        print("  --set -update Dataset-Labels")
      else:
        print("> Labels Updated:")
        print("\t- Label [%s]  ->  Label [%s]\n"
              "\t- Label [%s]  ->  Label [%s]\n"
              % (dataset_labels[0], default_label[0],
                dataset_labels[1], default_label[1]))

      dataset.iloc[:, label_col].replace(dataset_labels, default_label, inplace=True)
  else:
    if len(dataset_labels) < 2:
      print("Invalid dataset label count (Dataset Labels Below Required[2]).")
    elif len(dataset_labels) > 2:
      print("Invalid dataset label count (Dataset Labels Exceeds Max[2]).")
    return None, None

  return dataset, [[dataset_labels[0], default_label[0]],
                  [dataset_labels[1], default_label[1]]]


# Convert a np.ndarray to a pandas dataframe form
def pandas_dataframe(dataset):
  return pd.DataFrame(dataset)


# Convert a 1-D pandas dataframe into a vertical numpy of float values
def reshape_vertical_float(dataset):
  return dataset.values.reshape(-1, 1).astype(float)
