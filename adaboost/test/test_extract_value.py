from unittest import TestCase
from adaboost.common.get import extract_value

class TestSetParam(TestCase):

###############
# Boolean Cases
###############
# Arg Value
  def test_boolean_arg_value_case_0(self):
    testcase = True, False
    self.assertEqual(extract_value.arg_value(*testcase), None)


####################
# Number/float Cases
####################
  def test_number_arg_value_case_1(self):
    testcase = 1, -1
    self.assertEqual(extract_value.arg_value(*testcase), None)

  def test_number_arg_value_case_2(self):
    testcase = 3.14, -3.14
    self.assertEqual(extract_value.arg_value(*testcase), None)


############
# List Cases
############
  def test_list_arg_value_case_1(self):
    testcase = [0,1,2,3], [3,2,1,0]
    self.assertEqual(extract_value.arg_value(*testcase), None)


##############
# String Cases
##############
# Arg Value
  def test_string_arg_value_case_0(self):
    testcase = "somerandomstring", "some"
    self.assertEqual(extract_value.arg_value(*testcase), None)

  def test_string_arg_value_case_1(self):
    testcase = "some", "somerandomstring"
    self.assertEqual(extract_value.arg_value(*testcase), None)

  def test_string_arg_value_case_2(self):
    testcase = "application_value", "application_value=some_value"
    self.assertEqual(extract_value.arg_value(*testcase), "some_value")

  def test_string_arg_value_case_3(self):
    testcase = "application_value", "application_value=application_value_some_value"
    self.assertEqual(extract_value.arg_value(*testcase), "application_value_some_value")

  def test_string_arg_value_case_4(self):
    testcase = "application_value", "application_value-=some_value"
    self.assertEqual(extract_value.arg_value(*testcase), None)

  def test_string_arg_value_case_5(self):
    testcase = "application_value", "application_value=-some_value"
    self.assertEqual(extract_value.arg_value(*testcase), "-some_value")

  def test_string_arg_value_case_6(self):
    testcase = "application_value", "application_value-some_value"
    self.assertEqual(extract_value.arg_value(*testcase), None)

  def test_string_arg_value_case_7(self):
    testcase = "application_value", "application_value="
    self.assertEqual(extract_value.arg_value(*testcase), None)

  def test_string_arg_value_case_8(self):
    testcase = "", ""
    self.assertEqual(extract_value.arg_value(*testcase), None)

  if __name__ == '__main__':
    unittest.main()
