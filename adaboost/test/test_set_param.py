from unittest import TestCase
from adaboost.common import constants
from adaboost.common.set import set_param

class TestSetParam(TestCase):

  constants.initialize()

#########################
# Boolean Cases as String
#########################
# AdaBoost Estimators
  def test_boolean_adaboost_estimators_case_0(self):
    testcase = "True"
    self.assertEqual(set_param.adaboost_estimators(testcase), None)

  def test_boolean_adaboost_estimators_case_1(self):
    testcase = "False"
    self.assertEqual(set_param.adaboost_estimators(testcase), None)

  def test_boolean_adaboost_estimators_case_2(self):
    testcase = "true"
    self.assertEqual(set_param.adaboost_estimators(testcase), None)

  def test_boolean_adaboost_estimators_case_3(self):
    testcase = "false"
    self.assertEqual(set_param.adaboost_estimators(testcase), None)

  def test_boolean_adaboost_estimators_case_4(self):
    testcase = "trUe"
    self.assertEqual(set_param.adaboost_estimators(testcase), None)

  def test_boolean_adaboost_estimators_case_5(self):
    testcase = "fAlSe"
    self.assertEqual(set_param.adaboost_estimators(testcase), None)


# Dataset
  def test_boolean_dataset_case_0(self):
    testcase = "True"
    self.assertEqual(set_param.dataset(testcase), None)

  def test_boolean_dataset_case_1(self):
    testcase = "False"
    self.assertEqual(set_param.dataset(testcase), None)


# Dataset label column
  def test_boolean_dataset_label_col_case_0(self):
    testcase = "True", "False"
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_boolean_dataset_label_col_case_1(self):
    testcase = "False", "True"
    self.assertEqual(set_param.dataset_label_col(*testcase), None)


# Dataset feature columns
  def test_boolean_dataset_feature_cols_case_0(self):
    testcase = "True", "False"
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_boolean_dataset_feature_cols_case_1(self):
    testcase = "False", "True"
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)


# Dataset Sample Test Size
  def test_boolean_dataset_sample_test_size_case_0(self):
    testcase = "True", "True"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_boolean_dataset_sample_test_size_case_1(self):
    testcase = "tRuE", "TrUe"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_boolean_dataset_sample_test_size_case_2(self):
    testcase = "True", "False"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_boolean_dataset_sample_test_size_case_3(self):
    testcase = "False", "True"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_boolean_dataset_sample_test_size_case_4(self):
    testcase = "False", "False"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_boolean_dataset_sample_test_size_case_5(self):
    testcase = "FaLsE", "fAlSe"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)


# Out Detail
  def test_boolean_out_detail_case_0(self):
    testcase = "True"
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, True)

  def test_boolean_out_detail_case_1(self):
    testcase = "False"
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, False)

  def test_boolean_out_detail_case_2(self):
    testcase = "true"
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, True)

  def test_boolean_out_detail_case_3(self):
    testcase = "false"
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, False)

  def test_boolean_out_detail_case_4(self):
    testcase = "tRuE"
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, True)

  def test_boolean_out_detail_case_5(self):
    testcase = "FaLsE"
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, False)


# PCA Reduction
  def test_boolean_pca_reduction_case_0(self):
    testcase = "false", "true"
    self.assertEqual(set_param.pca_reduction(*testcase), None)

  def test_boolean_pca_reduction_case_1(self):
    testcase = "true", "false"
    self.assertEqual(set_param.pca_reduction(*testcase), None)


# Input SVM regularizer C
  def test_boolean_input_svm_regularizer_c_case_0(self):
    testcase = "False"
    self.assertEqual(set_param.svm_regularizer_c(testcase), None)

  def test_boolean_input_svm_regularizer_c_case_1(self):
    testcase = "True"
    self.assertEqual(set_param.svm_regularizer_c(testcase), None)


################################
# Number / Float Cases as String
################################
# AdaBoost Estimators
  def test_number_adaboost_estimators_case_0(self):
    testcase = "-1"
    self.assertEqual(set_param.adaboost_estimators(testcase), None)

  def test_number_adaboost_estimators_case_1(self):
    testcase = "0"
    self.assertEqual(set_param.adaboost_estimators(testcase), None)

  def test_number_adaboost_estimators_case_2(self):
    testcase = "10"
    self.assertEqual(set_param.adaboost_estimators(testcase), 10)

  def test_number_adaboost_estimators_case_3(self):
    testcase = "3.14"
    self.assertEqual(set_param.adaboost_estimators(testcase), None)


# Dataset
  def test_number_dataset_case_0(self):
    testcase = "-1"
    self.assertEqual(set_param.dataset(testcase), None)

  def test_number_dataset_case_1(self):
    testcase = "0"
    self.assertEqual(set_param.dataset(testcase), None)

  def test_number_dataset_case_2(self):
    testcase = "10"
    self.assertEqual(set_param.dataset(testcase), None)

  def test_number_dataset_case_3(self):
    testcase = "3.14"
    self.assertEqual(set_param.dataset(testcase), None)


# Dataset label column
  def test_number_dataset_label_col_case_0(self):
    testcase = "-1", "-1"
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_number_dataset_label_col_case_1(self):
    testcase = "-1", "1"
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_number_dataset_label_col_case_2(self):
    testcase = "0", "1"
    self.assertEqual(set_param.dataset_label_col(*testcase), 0)

  def test_number_dataset_label_col_case_3(self):
    testcase = "0", 1
    self.assertEqual(set_param.dataset_label_col(*testcase), 0)

  def test_number_dataset_label_col_case_4(self):
    testcase = "3.14", 1
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_number_dataset_label_col_case_5(self):
    testcase = "1", "3.14"
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_number_dataset_label_col_case_6(self):
    testcase = "3.14", "3.14"
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_number_dataset_label_col_case_7(self):
    testcase = "0", "0"
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_number_dataset_label_col_case_8(self):
    testcase = "10", "10"
    self.assertEqual(set_param.dataset_label_col(*testcase), 10)


# Dataset feature columns
  def test_number_dataset_feature_cols_case_0(self):
    testcase = "-1--1", "1"
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_number_dataset_feature_cols_case_1(self):
    testcase = "1--1", "1"
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_number_dataset_feature_cols_case_2(self):
    testcase = "1-1", "1"
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_number_dataset_feature_cols_case_3(self):
    testcase = "1-2", "2"
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, 1)
    self.assertEqual(test_value_end_col, 2)

  def test_number_dataset_feature_cols_case_4(self):
    testcase = "1-20", "10"
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_number_dataset_feature_cols_case_5(self):
    testcase = "1-20", 0
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_number_dataset_feature_cols_case_6(self):
    testcase = "0-0", 0
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_number_dataset_feature_cols_case_7(self):
    testcase = "1", 0
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_number_dataset_feature_cols_case_8(self):
    testcase = "3.14-3.5", 0
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)


# Dataset Sample Test Size
  def test_number_dataset_sample_test_size_case_0(self):
    testcase = "-1", "0"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_1(self):
    testcase = "0", "0"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_2(self):
    testcase = "10", "0"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_3(self):
    testcase = "3.14", "0"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_4(self):
    testcase = "-1", "-1"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_5(self):
    testcase = "0", "-1"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_6(self):
    testcase = "3.14", "-1"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_7(self):
    testcase = "10", "-1"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_8(self):
    testcase = "-1", "100"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_9(self):
    testcase = "0", "100"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_10(self):
    testcase = "100", "100"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_11(self):
    testcase = "101", "100"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_12(self):
    testcase = "10", "100"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, 10)
    self.assertEqual(test_value_test_size, 90)

  def test_number_dataset_sample_test_size_case_13(self):
    testcase = "3.14", "100"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_14(self):
    testcase = "0", "3.14"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_15(self):
    testcase = "1", "3.14"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_number_dataset_sample_test_size_case_16(self):
    testcase = "-1", "3.14"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)


# Out Detail
  def test_number_out_detail_case_0(self):
    testcase = "-1"
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, False)

  def test_number_out_detail_case_1(self):
    testcase = "0"
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, False)

  def test_number_out_detail_case_2(self):
    testcase = "10"
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, False)


# PCA Reduction
  def test_number_pca_reduction_case_0(self):
    testcase = "0", "0"
    self.assertEqual(set_param.pca_reduction(*testcase), "none")

  def test_number_pca_reduction_case_1(self):
    testcase = "1", "20"
    self.assertEqual(set_param.pca_reduction(*testcase), "default")

  def test_number_pca_reduction_case_2(self):
    testcase = "14", "20"
    self.assertEqual(set_param.pca_reduction(*testcase), 14)

  def test_number_pca_reduction_case_3(self):
    testcase = "25", "20"
    self.assertEqual(set_param.pca_reduction(*testcase), None)

  def test_number_pca_reduction_case_4(self):
    testcase = "1.0", "20"
    self.assertEqual(set_param.pca_reduction(*testcase), "default")

  def test_number_pca_reduction_case_5(self):
    testcase = "-1.0", "20"
    self.assertEqual(set_param.pca_reduction(*testcase), None)

  def test_number_pca_reduction_case_6(self):
    testcase = "0.0", "20"
    self.assertEqual(set_param.pca_reduction(*testcase), "none")

  def test_number_pca_reduction_case_7(self):
    testcase = "0.45", "20"
    self.assertEqual(set_param.pca_reduction(*testcase), 0.45)

  def test_number_pca_reduction_case_8(self):
    testcase = "1.45", "20"
    self.assertEqual(set_param.pca_reduction(*testcase), None)

  def test_number_pca_reduction_case_9(self):
    testcase = "-1.45", "20"
    self.assertEqual(set_param.pca_reduction(*testcase), None)


# Input SVM regularizer C
  def test_number_input_svm_regularizer_c_case_0(self):
    testcase = "0"
    self.assertEqual(set_param.svm_regularizer_c(testcase), 0.0)

  def test_number_input_svm_regularizer_c_case_1(self):
    testcase = "0.1"
    self.assertEqual(set_param.svm_regularizer_c(testcase), 0.1)

  def test_number_input_svm_regularizer_c_case_2(self):
    testcase = "1"
    self.assertEqual(set_param.svm_regularizer_c(testcase), 1.0)

  def test_number_input_svm_regularizer_c_case_3(self):
    testcase = "4"
    self.assertEqual(set_param.svm_regularizer_c(testcase), 4.0)

  def test_number_input_svm_regularizer_c_case_4(self):
    testcase = "3.14"
    self.assertEqual(set_param.svm_regularizer_c(testcase), 3.14)

  def test_number_input_svm_regularizer_c_case_5(self):
    testcase = "-3.14"
    self.assertEqual(set_param.svm_regularizer_c(testcase), None)

  def test_number_input_svm_regularizer_c_case_6(self):
    testcase = "-1"
    self.assertEqual(set_param.svm_regularizer_c(testcase), None)


######################
# List Cases as String
######################
# AdaBoost Input Estimators
  def test_list_adaboost_estimators_case_0(self):
    testcase = ["SomeTextThatShouldNotExists", "1", "3.14"]
    self.assertEqual(set_param.adaboost_estimators(testcase), None)


# Dataset
  def test_list_dataset_case_0(self):
    testcase = ["SomeTextThatShouldNotExists", "1", "3.14"]
    self.assertEqual(set_param.dataset(testcase), None)


# Dataset label column
  def test_list_dataset_label_col_case_0(self):
    testcase = "True", "False"
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_list_dataset_label_col_case_1(self):
    testcase = "False", "True"
    self.assertEqual(set_param.dataset_label_col(*testcase), None)


# Dataset feature columns
  def test_list_dataset_feature_cols_case_0(self):
    testcase = ["SomeTextThatShouldNotExists", "1", "3.14"], ["SomeTextThatShouldNotExists", "1", "3.14"]
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_list_dataset_feature_cols_case_1(self):
    testcase = ["SomeTextThatShouldNotExists", "1", "3.14"], ["SomeTextThatShouldNotExists", "1", "3.14"]
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)


# Dataset Sample Test Size
  def test_list_dataset_sample_test_size_case_0(self):
    testcase = ["SomeTextThatShouldNotExists", "1", "3.14"], ["SomeTextThatShouldNotExists", "1", "3.14"]
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)


# Out Detail
  def test_list_out_detail_case_0(self):
    testcase = ["SomeTextThatShouldNotExists", "1", "3.14"]
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, False)


# PCA Reduction
  def test_list_pca_reduction_case_0(self):
    testcase = ["SomeTextThatShouldNotExists", "1", "3.14"], ["SomeTextThatShouldNotExists", "1", "3.14"]
    self.assertEqual(set_param.pca_reduction(*testcase), None)


# Input SVM regularizer C
  def test_list_input_svm_regularizer_c_case_0(self):
    testcase = ["SomeTextThatShouldNotExists", "1", "3.14"]
    self.assertEqual(set_param.svm_regularizer_c(testcase), None)

##############
# String Cases
##############
# AdaBoost Input Estimators
  def test_string_adaboost_estimators_case_0(self):
    testcase = "SomeTextThatShouldNotExists"
    self.assertEqual(set_param.adaboost_estimators(testcase), None)


# Dataset
  def test_string_dataset_case_0(self):
    testcase = "default"
    self.assertEqual(set_param.dataset(testcase), constants.WDBC_DATASET_PATH)

  def test_string_dataset_case_1(self):
    testcase = "default1"
    self.assertEqual(set_param.dataset(testcase), None)

  def test_string_dataset_case_2(self):
    testcase = "./data/raw/wdbc_data.csv"
    self.assertEqual(set_param.dataset(testcase), "./data/raw/wdbc_data.csv")

  def test_string_dataset_case_3(self):
    testcase = "./data/fake/wdbc_data.csv"
    self.assertEqual(set_param.dataset(testcase), None)


# Dataset label column
  def test_string_dataset_label_col_case_0(self):
    testcase = "SomeNonExistantTest", "1"
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_string_dataset_label_col_case_1(self):
    testcase = "SomeNonExistantTest", 1
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_string_dataset_label_col_case_2(self):
    testcase = "SomeNonExistantTest", -1
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_string_dataset_label_col_case_3(self):
    testcase = "SomeNonExistantTest", 10
    self.assertEqual(set_param.dataset_label_col(*testcase), None)

  def test_string_dataset_label_col_case_4(self):
    testcase = "SomeNonExistantTest", 0
    self.assertEqual(set_param.dataset_label_col(*testcase), None)


# Dataset feature columns
  def test_string_dataset_feature_cols_case_0(self):
    testcase = "SomeNonExistantTest", "1"
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_string_dataset_feature_cols_case_1(self):
    testcase = "SomeNonExistantTest", 1
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_string_dataset_feature_cols_case_2(self):
    testcase = "SomeNonExistantTest", -1
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_string_dataset_feature_cols_case_3(self):
    testcase = "SomeNonExistantTest", 10
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)

  def test_string_dataset_feature_cols_case_4(self):
    testcase = "SomeNonExistantTest", 0
    test_value_start_col, test_value_end_col = set_param.dataset_feature_cols(*testcase)
    self.assertEqual(test_value_start_col, None)
    self.assertEqual(test_value_end_col, None)


# Dataset Sample Test Size
  def test_string_dataset_sample_test_size_case_0(self):
    testcase = "SomeTextThatShouldNotExists", "0"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_string_dataset_sample_test_size_case_1(self):
    testcase = "SomeTextThatShouldNotExists", "-1"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_string_dataset_sample_test_size_case_2(self):
    testcase = "SomeTextThatShouldNotExists", "100"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_string_dataset_sample_test_size_case_3(self):
    testcase = "0", "SomeTextThatShouldNotExists"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_string_dataset_sample_test_size_case_4(self):
    testcase = "-1", "SomeTextThatShouldNotExists"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_string_dataset_sample_test_size_case_5(self):
    testcase = "100", "SomeTextThatShouldNotExists"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)

  def test_string_dataset_sample_test_size_case_6(self):
    testcase = "SomeTextThatShouldNotExists", "SomeTextThatShouldNotExists"
    test_value_sample_size, test_value_test_size = set_param.dataset_sample_test_size(*testcase)
    self.assertEqual(test_value_sample_size, None)
    self.assertEqual(test_value_test_size, None)


# Out Detail
  def test_boolean_out_detail_case_0(self):
    testcase = "SomeTextThatShouldNotExists"
    set_param.out_detail(testcase)
    self.assertEqual(constants.OUTPUT_DETAIL, False)


# PCA Reduction
  def test_string_pca_reduction_case_0(self):
    testcase = "none", "none"
    self.assertEqual(set_param.pca_reduction(*testcase), "none")

  def test_string_pca_reduction_case_1(self):
    testcase = "default", "default"
    self.assertEqual(set_param.pca_reduction(*testcase), "default")

  def test_string_pca_reduction_case_2(self):
    testcase = "default", 4
    self.assertEqual(set_param.pca_reduction(*testcase), "default")

  def test_string_pca_reduction_case_2(self):
    testcase = "something that will fail", 4
    self.assertEqual(set_param.pca_reduction(*testcase), None)


# Input SVM regularizer C
  def test_string_input_svm_regularizer_c_case_0(self):
    testcase = "none"
    self.assertEqual(set_param.svm_regularizer_c(testcase), 0.0)

  def test_string_input_svm_regularizer_c_case_1(self):
    testcase = "default"
    self.assertEqual(set_param.svm_regularizer_c(testcase), 1.0)

  def test_string_input_svm_regularizer_c_case_2(self):
    testcase = "sometest"
    self.assertEqual(set_param.svm_regularizer_c(testcase), None)

if __name__ == '__main__':
  unittest.main()
