from .convert_type import (
  to_bool,
  to_float,
  to_int,
  to_string
)

from .format_dataset import (
  dataset_extract_columns,
  dataset_extract_features,
  dataset_default_label,
  pandas_dataframe,
  reshape_vertical_float
)

from .classification import (
  svm_prediction,
  prediction_comparison,
  prediction_accuracy
)