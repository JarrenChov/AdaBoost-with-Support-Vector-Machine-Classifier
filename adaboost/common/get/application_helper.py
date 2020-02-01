# Print argument parameters for passing as command line when running application
def application_help_details():
  print("## Application Name: AdaBoost-with-Support-Vector-Machine-Classifier")
  print("## Last Modified Date: Tuesday, January 28 2020")

  print()

  print("## Parameter Arguments\n"
        "--------------------------------------------------------------\n"
        "Argument Requirements:\n"
        " \u00ac [*] - Denotes a required parameter.\n"
        " \u00ac [!*] - denotes a required parameter given that the default dataset is not used.\n"
        " \u00ac [#] - Denotes a optional parameter, where a default value will be overrided if such parameter is supplied.\n\n"

        "\n"

        "  [*] dataset_file=<value> - specifies a dataset file to be used or use default supplied file.\n"
        "      <value> can accept either of the following:\n"
        "\t \u25cf default_# - Uses the supplied datasets present, where # corresponds to the datasets below:\n"
        "\t\t \u00d7 ( 1 ) Wisconsin Diagnostic Breast Cancer (WDBC) Dataset\n"
        "\t\t \u00d7 ( 2 ) Unamed Dataset by Chunhua Shen [Subset sample - 6000 samples]\n"
        "\t\t \u00d7 ( 3 ) Unamed Dataset by Chunhua Shen [Full sample - 10000 samples]\n"
        "\t \u25cf Any string path of a file-path leading to a dataset file.\n"
        "\t   Note: [Directory starts at the root using relative path, .csv format required (type not checked)].\n\n"

        "  [*] dataset_sample_size=<value> - specifies a size to split the dataset into a training and testing set.\n"
        "      <value> can accept either of the following:\n"
        "\t \u25cf A string integer value denoting a number value.\n\n"

        "  [!*] dataset_label_column=<value> - specifies the column number for the dataset labels.\n"
        "      <value> can accept either of the following:\n"
        "\t \u25cf A integer value denoting a column index in a dataset .\n"
        "\t   Note: [Column index start at the 0th index].\n\n"

        "  [!*] dataset_feature_columns=<value>-<value> - specifies the range of columns dataset features span.\n"
        "      <value> can accept either of the following:\n"
        "\t \u25cf A integer value denoting a column index in a dataset .\n"
        "\t   Note: [Column index start at the 0th index].\n\n"

        "  [#] pca_reduction=<value> - specifies the size to reduce an existing dataset features to.\n"
        "      \u00BA Unspecified <value> will revert to default reduction string <value=none>\n"
        "      <value> can accept either of the following:\n"
        "\t \u25cf A string value {default | none} denoting either a default or no reduction to dataset.\n"
        "\t \u25cf A float value in the range of {0 <-> 1} denoting a proportional reduction size to dataset.\n"
        "\t \u25cf A integer value denoting a subset of a dataset.\n\n"

        "  [*] svm_regularizer_c=<value> - specifies the boundary in misclassified points for a non-linearly separable dataset.\n"
        "      \u00BA A greater C values generates a more complex and tailored boundary to the data,\n"
        "        whilst a lower C values generates a more generalized boundary.\n"
        "      <value> can accept either of the following:\n"
        "\t \u25cf A string value {default | none} denoting either a default boundary C=1.0 (soft-margin case),\n"
        "\t   or cases in which all points lies on respective sides of a boundary (hard-margin case).\n"
        "\t \u25cf A float value denoting the boundary complexity and mis-classification of points.\n\n"

        "  [*] adaboost_estimators=<value> - specifies the number of AdaBoost generated prediction models (weak learners).\n"
        "      <value> can accept either of the following:\n"
        "\t \u25cf A string integer value denoting a number value.\n\n"

        "  [#] output_detail=<value> - specifies verbose printing.\n"
        "      \u00BA Unspecified <value> will revert to default boolean <value=false>\n"
        "      <value> can accept either of the following:\n"
        "\t \u25cf A string boolean value {true | false} denoting a state.\n"
        "==============================================================")

  return 0
