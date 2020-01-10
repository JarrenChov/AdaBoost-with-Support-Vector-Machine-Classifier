from unittest import mock, skip, TestCase
from adaboost.common import constants
from adaboost.common.get import retrieve_param

class TestRetrieveParam(TestCase):
# Notes:
# =============================================================================
#   Some cases may have a skip, this is intentional as no output is produced
#   due to a while loop condition until an acceptable response is obtained.
#
# The following cases and its identifier detail the information corresponding to its information.
# ADABOOST_NEG_NUM - A negative number was supplied, such that only numbers > 0 are accepted.
# ADABOOST_ZERO_NUM - A number 0 was supplied, such that only numbers > 0 are accepted.
# ADABOOST_EMP_STRING - A empty string was supplied, such that only a non-empty string
#                       or a number is accepted.
# ADABOOST_ERR_IN_NUM - A invalid input, such that only numbers (integers) are accepted.
# ADABOOST_ERR_IN_STRING - A invalid input, such that only strings are accepted.
# DATASET_NEG_NUM - A negative number was supplied, such that only '1' is accepted.
# DATASET_ZERO_NUM - A number 0 was supplied, such that only '1' is accepted.
# DATASET_OVR_NUM - A number was supplied but no such dataset exists, only '1' is accepted.
# OUT_DETAIL_IN_STR - A string was obtained, but string was invalid.
# =============================================================================

  constants.initialize()

###############################################
# Number / Float Cases as String (Python Input)
###############################################
# AdaBoost Input Estimators
  @skip("ADABOOST_NEG_NUM")
  def test_number_adaboost_estimators_case_0(self):
    with mock.patch('builtins.input', return_value="-1"):
      self.assertEqual(retrieve_param.adaboost_input_estimators(), None)

  @skip("ADABOOST_ZERO_NUM")
  def test_number_adaboost_estimators_case_1(self):
    with mock.patch('builtins.input', return_value="0"):
      self.assertEqual(retrieve_param.adaboost_input_estimators(), None)

  @skip("ADABOOST_ERR_IN_NUM")
  def test_number_adaboost_estimators_case_2(self):
    with mock.patch('builtins.input', return_value="3.14"):
      self.assertEqual(retrieve_param.adaboost_input_estimators(), None)

  def test_number_adaboost_estimators_case_3(self):
    with mock.patch('builtins.input', return_value="1"):
      self.assertEqual(retrieve_param.adaboost_input_estimators(), "1")

  def test_number_adaboost_estimators_case_4(self):
    with mock.patch('builtins.input', return_value="400"):
      self.assertEqual(retrieve_param.adaboost_input_estimators(), "400")


# Input Dataset
  @skip("DATASET_OVR_NUM")
  def test_number_dataset_case_0(self):
   with mock.patch('builtins.input', return_value=40):
      self.assertEqual(retrieve_param.input_dataset(), None)

  @skip("DATASET_NEG_NUM")
  def test_number_dataset_case_1(self):
   with mock.patch('builtins.input', return_value=-1):
      self.assertEqual(retrieve_param.input_dataset(), None)

  @skip("ADABOOST_ERR_IN_NUM")
  def test_number_dataset_case_2(self):
    with mock.patch('builtins.input', return_value="3.14"):
      self.assertEqual(retrieve_param.input_dataset(), None)

  def test_number_dataset_case_3(self):
    with mock.patch('builtins.input', return_value="1"):
      self.assertEqual(retrieve_param.input_dataset(), "default")


# Input Dataset Sample Size
  @skip("ADABOOST_NEG_NUM")
  def test_number_dataset_sample_size_case_0(self):
    with mock.patch('builtins.input', return_value="-1"):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), None)

  @skip("ADABOOST_ZERO_NUM")
  def test_number_dataset_sample_size_case_1(self):
    with mock.patch('builtins.input', return_value="0"):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), None)

  @skip("ADABOOST_ERR_IN_NUM")
  def test_number_dataset_sample_size_case_2(self):
    with mock.patch('builtins.input', return_value="3.14"):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), None)

  def test_number_dataset_sample_size_case_3(self):
    with mock.patch('builtins.input', return_value="1"):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), "1")

  def test_number_dataset_sample_size_case_4(self):
    with mock.patch('builtins.input', return_value="400"):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), "400")


# Input Out Details
  @skip("ADABOOST_ERR_IN_STRING")
  def test_number_out_detail_case_0(self):
    with mock.patch('builtins.input', return_value="1"):
      self.assertEqual(retrieve_param.input_out_detail(), None)

  @skip("ADABOOST_ERR_IN_STRING")
  def test_number__out_detail_case_1(self):
    with mock.patch('builtins.input', return_value="-100"):
      self.assertEqual(retrieve_param.input_out_detail(), None)


##############
# String Cases
##############
# AdaBoost Input Estimators
  @skip("ADABOOST_EMP_STRING")
  def test_string_adaboost_estimators_case_0(self):
    with mock.patch('builtins.input', return_value=''):
      self.assertEqual(retrieve_param.adaboost_input_estimators(), None)

  @skip("ADABOOST_ERR_IN_STRING")
  def test_string_adaboost_estimators_case_1(self):
    with mock.patch('builtins.input', return_value="some string"):
      self.assertEqual(retrieve_param.adaboost_input_estimators(), None)


# Input Dataset
  def test_string_dataset_case_0(self):
    with mock.patch('builtins.input', return_value="some purposely fail input, captured in the set func"):
      self.assertEqual(retrieve_param.input_dataset(), "some purposely fail input, captured in the set func")

  def test_string_dataset_case_1(self):
   with mock.patch('builtins.input', return_value="./File/some_loc/dataset/file.csv"):
      self.assertEqual(retrieve_param.input_dataset(), "./File/some_loc/dataset/file.csv")

  @skip("ADABOOST_EMP_STRING")
  def test_string_dataset_case_2(self):
   with mock.patch('builtins.input', return_value=''):
      self.assertEqual(retrieve_param.input_dataset(), None)

  def test_string_dataset_case_3(self):
    with mock.patch('builtins.input', side_effect = [" test", " test ", "test "]):
      self.assertEqual(retrieve_param.input_dataset(), "test")
      self.assertEqual(retrieve_param.input_dataset(), "test")
      self.assertEqual(retrieve_param.input_dataset(), "test")


# Input Dataset Sample Size
  @skip("ADABOOST_EMP_STRING")
  def test_string_dataset_sample_size_case_0(self):
    with mock.patch('builtins.input', return_value=''):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), None)

  @skip("ADABOOST_ERR_IN_STRING")
  def test_string_dataset_sample_size_case_1(self):
    with mock.patch('builtins.input', return_value="some string"):
      self.assertEqual(retrieve_param.input_dataset_sample_size(), None)


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

  @skip("OUT_DETAIL_IN_STR")
  def test_string_out_detail_case_4(self):
    with mock.patch('builtins.input', return_value="some string"):
      self.assertEqual(retrieve_param.input_out_detail(), None)

if __name__ == '__main__':
  unittest.main()
