# Initialize global variables
def initialize():
  # Declaration of default label of first class
  global DEFAULT_LABEL_0
  DEFAULT_LABEL_0 = -1


  # Declaration of default label of second class
  global DEFAULT_LABEL_1
  DEFAULT_LABEL_1 = 1


  # Declaration of default WDBC dataset
  global WDBC_DATASET_PATH
  WDBC_DATASET_PATH = "./data/wdbc/raw/wdbc_data.csv"


  # Declaration of default WDBC dataset label column
  global WDBC_DATASET_LABEL_COL
  WDBC_DATASET_LABEL_COL = 1


  # Declaration of default WDBC dataset feature start column
  global WDBC_DATASET_FEATURE_START_COL
  WDBC_DATASET_FEATURE_START_COL = 2


  # Declaration of default WDBC dataset feature end column
  global WDBC_DATASET_FEATURE_END_COL
  WDBC_DATASET_FEATURE_END_COL = 32


  # Declaration of default SHEN dataset
  global SHEN_DATASET_SUBSET_PATH
  SHEN_DATASET_SUBSET_PATH = "./data/unamed/processed/chunhua_shen_6000.csv"


  # Declaration of default SHEN dataset
  global SHEN_DATASET_FULL_PATH
  SHEN_DATASET_FULL_PATH = "./data/unamed/processed/chunhua_shen_10000.csv"


  # Declaration of default SHEN dataset label column
  global SHEN_DATASET_LABEL_COL
  SHEN_DATASET_LABEL_COL = 0


  # Declaration of default SHEN dataset feature start column
  global SHEN_DATASET_FEATURE_START_COL
  SHEN_DATASET_FEATURE_START_COL = 1


  # Declaration of default SHEN dataset feature end column
  global SHEN_DATASET_FEATURE_END_COL
  SHEN_DATASET_FEATURE_END_COL = 201


  # Declaration of SVM threshold for non-zero lagrange multiplier
  global NON_ZERO_LAGRANGE_THRESHOLD
  NON_ZERO_LAGRANGE_THRESHOLD = 1e-5


  # Declaration of SVM threshold for non-zero lagrange multiplier
  global ZERO_SIGNIFICANCE_THRESHOLD
  ZERO_SIGNIFICANCE_THRESHOLD = 1e-10


  # Declaration of detailed output printing
  global OUTPUT_DETAIL
  OUTPUT_DETAIL = False
