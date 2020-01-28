from unittest import mock, TestCase
from adaboost.common import constants
from adaboost.common.get import retrieve_param

class TestRetrieveParam(TestCase):

  constants.initialize()

###############################################
# Number / Float Cases as String (Python Input)
###############################################
# AdaBoost Input Estimators
  def test_number_adaboost_estimators_case_0(self):
    with mock.patch('builtins.input', side_effect = ["-1", "1"]):
      self.assertEqual(retrieve_param.input_adaboost_estimators(), "1")

  def test_number_adaboost_estimators_case_1(self):
    with mock.patch('builtins.input', side_effect = ["0", "1"]):
      self.assertEqual(retrieve_param.input_adaboost_estimators(), "1")

  def test_number_adaboost_estimators_case_2(self):
    with mock.patch('builtins.input', side_effect = ["3.14", "1"]):
      self.assertEqual(retrieve_param.input_adaboost_estimators(), "1")

  def test_number_adaboost_estimators_case_3(self):
    with mock.patch('builtins.input', return_value="1"):
      self.assertEqual(retrieve_param.input_adaboost_estimators(), "1")

  def test_number_adaboost_estimators_case_4(self):
    with mock.patch('builtins.input', return_value="400"):
      self.assertEqual(retrieve_param.input_adaboost_estimators(), "400")


# Input Dataset
  def test_number_input_dataset_case_0(self):
   with mock.patch('builtins.input', side_effect = ["40", "1"]):
      self.assertEqual(retrieve_param.input_dataset(), "default_1")

  def test_number_input_dataset_case_1(self):
   with mock.patch('builtins.input', side_effect = ["-1", "1"]):
      self.assertEqual(retrieve_param.input_dataset(), "default_1")

  def test_number_input_dataset_case_2(self):
    with mock.patch('builtins.input', side_effect = ["3.14", "1"]):
      self.assertEqual(retrieve_param.input_dataset(), "default_1")

  def test_number_input_dataset_case_3(self):
    with mock.patch('builtins.input', return_value="1"):
      self.assertEqual(retrieve_param.input_dataset(), "default_1")

  def test_number_input_dataset_case_4(self):
    with mock.patch('builtins.input', return_value="2"):
      self.assertEqual(retrieve_param.input_dataset(), "default_2")

  def test_number_input_dataset_case_5(self):
    with mock.patch('builtins.input', return_value="3"):
      self.assertEqual(retrieve_param.input_dataset(), "default_3")


# Input Dataset Sample Size
  def test_number_dataset_sample_size_case_0(self):
    with mock.patch('builtins.input', side_effect = ["-1", "1"]):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), "1")

  def test_number_dataset_sample_size_case_1(self):
    with mock.patch('builtins.input', side_effect = ["0", "1"]):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), "1")

  def test_number_dataset_sample_size_case_2(self):
    with mock.patch('builtins.input', side_effect = ["3.14", "1"]):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), "1")

  def test_number_dataset_sample_size_case_3(self):
    with mock.patch('builtins.input', return_value="1"):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), "1")

  def test_number_dataset_sample_size_case_4(self):
    with mock.patch('builtins.input', return_value="400"):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), "400")


# Input_dataset Feature COlumn
  def test_number_input_dataset_features_col_case_0(self):
    with mock.patch('builtins.input', side_effect = ["1-1", "1-2"]):
      self.assertEqual(retrieve_param.input_dataset_features_col(), "1-2")

  def test_number_input_dataset_features_col_case_1(self):
    with mock.patch('builtins.input', side_effect = ["-1-1", "0-2"]):
      self.assertEqual(retrieve_param.input_dataset_features_col(), "0-2")

  def test_number_input_dataset_features_col_case_2(self):
    with mock.patch('builtins.input', side_effect = ["20-10", "10-20"]):
      self.assertEqual(retrieve_param.input_dataset_features_col(), "10-20")

  def test_number_input_dataset_features_col_case_3(self):
    with mock.patch('builtins.input', side_effect = ["1--2", "1-2"]):
      self.assertEqual(retrieve_param.input_dataset_features_col(), "1-2")

  def test_number_input_dataset_features_col_case_4(self):
    with mock.patch('builtins.input', side_effect = ["1.5--2.5", "1-2"]):
      self.assertEqual(retrieve_param.input_dataset_features_col(), "1-2")


# Input_dataset Label Column
  def test_number_input_dataset_label_col_case_0(self):
    with mock.patch('builtins.input', side_effect = ["3.14", "1"]):
      self.assertEqual(retrieve_param.input_dataset_label_col(), "1")

  def test_number_input_dataset_label_col_case_1(self):
    with mock.patch('builtins.input', return_value="0"):
      self.assertEqual(retrieve_param.input_dataset_label_col(), "0")

  def test_number_input_dataset_label_col_case_2(self):
    with mock.patch('builtins.input', side_effect = ["-1", "1"]):
      self.assertEqual(retrieve_param.input_dataset_label_col(), "1")

  def test_number_input_dataset_label_col_case_3(self):
    with mock.patch('builtins.input', return_value="100"):
      self.assertEqual(retrieve_param.input_dataset_label_col(), "100")


# Input Out Details
  def test_number_out_detail_case_0(self):
    with mock.patch('builtins.input', side_effect = ["1", "n"]):
      self.assertEqual(retrieve_param.input_out_detail(), "false")

  def test_number__out_detail_case_1(self):
    with mock.patch('builtins.input', side_effect = ["-100", "yes"]):
      self.assertEqual(retrieve_param.input_out_detail(), "true")


# PCA input reduction
  def test_number_input_pca_reduction_case_0(self):
    with mock.patch('builtins.input', side_effect = ["3.14", "1"]):
      self.assertEqual(retrieve_param.input_pca_reduction(), "1")

  def test_number_input_pca_reduction_case_1(self):
    with mock.patch('builtins.input', side_effect = ["3.14", "0"]):
      self.assertEqual(retrieve_param.input_pca_reduction(), "0")

  def test_number_input_pca_reduction_case_2(self):
    with mock.patch('builtins.input', side_effect = ["3.14", "0.45"]):
      self.assertEqual(retrieve_param.input_pca_reduction(), "0.45")

  def test_number_input_pca_reduction_case_4(self):
    with mock.patch('builtins.input', side_effect = ["3.14", "-1", "0.45"]):
      self.assertEqual(retrieve_param.input_pca_reduction(), "0.45")

  def test_number_input_pca_reduction_case_5(self):
    with mock.patch('builtins.input', return_value="1"):
      self.assertEqual(retrieve_param.input_pca_reduction(), "1")

  def test_number_input_pca_reduction_case_6(self):
    with mock.patch('builtins.input', return_value="20"):
      self.assertEqual(retrieve_param.input_pca_reduction(), "20")


# Input SVM regularizer C
  def test_number_input_svm_regularizer_c_case_0(self):
    with mock.patch('builtins.input', side_effect = ["-1", "1"]):
      self.assertEqual(retrieve_param.input_svm_regularizer_c(), "default")

  def test_number_input_svm_regularizer_c_case_1(self):
    with mock.patch('builtins.input', return_value="0"):
      self.assertEqual(retrieve_param.input_svm_regularizer_c(), "none")

  def test_number_input_svm_regularizer_c_case_2(self):
    with mock.patch('builtins.input', return_value="3.14"):
      self.assertEqual(retrieve_param.input_svm_regularizer_c(), "3.14")

  def test_number_input_svm_regularizer_c_case_3(self):
    with mock.patch('builtins.input', return_value="1"):
      self.assertEqual(retrieve_param.input_svm_regularizer_c(), "default")

  def test_number_input_svm_regularizer_c_case_4(self):
    with mock.patch('builtins.input', return_value="400"):
      self.assertEqual(retrieve_param.input_svm_regularizer_c(), "400")



##############
# String Cases
##############
# AdaBoost Input Estimators
  def test_string_adaboost_estimators_case_0(self):
    with mock.patch('builtins.input', side_effect = ['', "1"]):
      self.assertEqual(retrieve_param.input_adaboost_estimators(), "1")

  def test_string_adaboost_estimators_case_1(self):
    with mock.patch('builtins.input', side_effect = ["some string", "1"]):
      self.assertEqual(retrieve_param.input_adaboost_estimators(), "1")


# Input Dataset
  def test_string_input_dataset_case_0(self):
    with mock.patch('builtins.input', return_value="some purposely fail input, captured in the set func"):
      self.assertEqual(retrieve_param.input_dataset(), "some purposely fail input, captured in the set func")

  def test_string_input_dataset_case_1(self):
   with mock.patch('builtins.input', return_value="./File/some_loc/dataset/file.csv"):
      self.assertEqual(retrieve_param.input_dataset(), "./File/some_loc/dataset/file.csv")

  def test_string_input_dataset_case_2(self):
   with mock.patch('builtins.input', side_effect = ['', "some string!"]):
      self.assertEqual(retrieve_param.input_dataset(), "some string!")

  def test_string_input_dataset_case_3(self):
    with mock.patch('builtins.input', side_effect = [" test", " test ", "test "]):
      self.assertEqual(retrieve_param.input_dataset(), "test")
      self.assertEqual(retrieve_param.input_dataset(), "test")
      self.assertEqual(retrieve_param.input_dataset(), "test")


# Input Dataset Sample Size
  def test_string_dataset_sample_size_case_0(self):
    with mock.patch('builtins.input', side_effect = ['', "1"]):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), "1")

  def test_string_dataset_sample_size_case_1(self):
    with mock.patch('builtins.input', side_effect = ["some string", "1"]):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), "1")


# Input Out Details
  def test_string_out_detail_case_0(self):
    with mock.patch('builtins.input', return_value="yes"):
      self.assertEqual(retrieve_param.input_out_detail(), "true")

  def test_string_out_detail_case_1(self):
    with mock.patch('builtins.input', return_value="y"):
      self.assertEqual(retrieve_param.input_out_detail(), "true")

  def test_string_out_detail_case_2(self):
    with mock.patch('builtins.input', return_value="no"):
      self.assertEqual(retrieve_param.input_out_detail(), "false")

  def test_string_out_detail_case_3(self):
    with mock.patch('builtins.input', return_value="n"):
      self.assertEqual(retrieve_param.input_out_detail(), "false")

  def test_string_out_detail_case_4(self):
    with mock.patch('builtins.input',side_effect = ["some string", "Yes"]):
      self.assertEqual(retrieve_param.input_out_detail(), "true")


# PCA input reduction
  def test_string_input_pca_reduction_case_0(self):
    with mock.patch('builtins.input', return_value="nOne"):
      self.assertEqual(retrieve_param.input_pca_reduction(), "none")

  def test_string_input_pca_reduction_case_1(self):
    with mock.patch('builtins.input', return_value="NONE"):
      self.assertEqual(retrieve_param.input_pca_reduction(), "none")

  def test_string_input_pca_reduction_case_2(self):
    with mock.patch('builtins.input', return_value="None"):
      self.assertEqual(retrieve_param.input_pca_reduction(), "none")

  def test_string_input_pca_reduction_case_3(self):
    with mock.patch('builtins.input', return_value="defaulT"):
      self.assertEqual(retrieve_param.input_pca_reduction(), "default")

  def test_string_input_pca_reduction_case_4(self):
    with mock.patch('builtins.input', return_value="DEFAULT"):
      self.assertEqual(retrieve_param.input_pca_reduction(), "default")

  def test_string_input_pca_reduction_case_5(self):
    with mock.patch('builtins.input', return_value="default"):
      self.assertEqual(retrieve_param.input_pca_reduction(), "default")

  def test_string_input_pca_reduction_case_6(self):
    with mock.patch('builtins.input', side_effect = ["3.14", "none"]):
      self.assertEqual(retrieve_param.input_pca_reduction(), "none")


# Input SVM regularizer C
  def test_string_input_svm_regularizer_c_case_0(self):
    with mock.patch('builtins.input', side_effect = ["aaa", "DeFaUlT"]):
      self.assertEqual(retrieve_param.input_svm_regularizer_c(), "default")

  def test_string_input_svm_regularizer_c_case_1(self):
    with mock.patch('builtins.input', return_value="NONE"):
      self.assertEqual(retrieve_param.input_svm_regularizer_c(), "none")

  def test_string_input_svm_regularizer_c_case_2(self):
    with mock.patch('builtins.input', return_value="DEFAULT"):
      self.assertEqual(retrieve_param.input_svm_regularizer_c(), "default")

if __name__ == '__main__':
  unittest.main()
