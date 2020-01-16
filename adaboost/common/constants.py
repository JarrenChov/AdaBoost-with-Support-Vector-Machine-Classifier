# Initialize global variables
def initialize():
  # Declaration of default WDBC dataset
  global WDBC_DATASET_PATH
  WDBC_DATASET_PATH = "./data/raw/wdbc_data.csv"


  # Declaration of malignant label
  global MALIGNANT_LABEL
  MALIGNANT_LABEL = 1


  # Declaration of benign label
  global BENIGN_LABEL
  BENIGN_LABEL = -1


  # Declaration of detailed output printing
  global OUTPUT_DETAIL
  OUTPUT_DETAIL = False
