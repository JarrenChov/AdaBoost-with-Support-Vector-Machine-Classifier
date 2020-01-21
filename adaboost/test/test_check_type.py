from unittest import TestCase
from adaboost.common.check import check_type

class TestCheckType(TestCase):
###############
# Boolean Cases
###############
# Boolean Cases
  def test_boolean_is_boolean_case_0(self):
    testcase = True
    self.assertEqual(check_type.is_bool(testcase), True)

  def test_boolean_is_boolean_case_1(self):
    testcase = False
    self.assertEqual(check_type.is_bool(testcase), True)


# Float Cases
  def test_boolean_is_float_case_0(self):
    testcase = True
    self.assertEqual(check_type.is_float(testcase), False)

  def test_boolean_is_float_case_1(self):
    testcase = False
    self.assertEqual(check_type.is_float(testcase), False)


# Integer Cases
  def test_boolean_is_int_case_0(self):
    testcase = True
    self.assertEqual(check_type.is_int(testcase), True)

  def test_boolean_is_int_case_1(self):
    testcase = False
    self.assertEqual(check_type.is_int(testcase), True)


# List Cases
  def test_boolean_is_list_case_0(self):
    testcase = True
    self.assertEqual(check_type.is_list(testcase), False)

  def test_boolean_is_list_case_1(self):
    testcase = False
    self.assertEqual(check_type.is_list(testcase), False)


# String Cases
  def test_boolean_is_str_case_0(self):
    testcase = True
    self.assertEqual(check_type.is_str(testcase), False)

  def test_boolean_is_str_case_1(self):
    testcase = False
    self.assertEqual(check_type.is_str(testcase), False)


######################
# Number / Float Cases
######################
# Boolean Cases
  def test_number_is_boolean_case_0(self):
    testcase = 1
    self.assertEqual(check_type.is_bool(testcase), False)

  def test_number_is_boolean_case_1(self):
    testcase = -1
    self.assertEqual(check_type.is_bool(testcase), False)

  def test_number_is_boolean_case_2(self):
    testcase = 0
    self.assertEqual(check_type.is_bool(testcase), False)

  def test_number_is_boolean_case_3(self):
    testcase = 3.14
    self.assertEqual(check_type.is_bool(testcase), False)

  def test_number_is_boolean_case_4(self):
    testcase = -3.14
    self.assertEqual(check_type.is_bool(testcase), False)


# Float Cases
  def test_number_is_float_case_0(self):
    testcase = 1
    self.assertEqual(check_type.is_float(testcase), False)

  def test_number_is_float_case_1(self):
    testcase = -1
    self.assertEqual(check_type.is_float(testcase), False)

  def test_number_is_float_case_2(self):
    testcase = 0
    self.assertEqual(check_type.is_float(testcase), False)

  def test_number_is_float_case_3(self):
    testcase = 3.14
    self.assertEqual(check_type.is_float(testcase), True)

  def test_number_is_float_case_4(self):
    testcase = -3.14
    self.assertEqual(check_type.is_float(testcase), True)


# Integer Cases
  def test_number_is_int_case_0(self):
    testcase = 1
    self.assertEqual(check_type.is_int(testcase), True)

  def test_number_is_int_case_1(self):
    testcase = -1
    self.assertEqual(check_type.is_int(testcase), True)

  def test_number_is_int_case_2(self):
    testcase = 0
    self.assertEqual(check_type.is_int(testcase), True)

  def test_number_is_int_case_3(self):
    testcase = 3.14
    self.assertEqual(check_type.is_int(testcase), False)

  def test_number_is_int_case_4(self):
    testcase = -3.14
    self.assertEqual(check_type.is_int(testcase), False)


# List Cases
  def test_number_is_list_case_0(self):
    testcase = 1
    self.assertEqual(check_type.is_list(testcase), False)

  def test_number_is_list_case_1(self):
    testcase = -1
    self.assertEqual(check_type.is_list(testcase), False)

  def test_number_is_list_case_2(self):
    testcase = 0
    self.assertEqual(check_type.is_list(testcase), False)

  def test_number_is_list_case_3(self):
    testcase = 3.14
    self.assertEqual(check_type.is_list(testcase), False)

  def test_number_is_list_case_4(self):
    testcase = -3.14
    self.assertEqual(check_type.is_list(testcase), False)


# String Cases
  def test_number_is_str_case_0(self):
    testcase = 1
    self.assertEqual(check_type.is_str(testcase), False)

  def test_number_is_str_case_1(self):
    testcase = -1
    self.assertEqual(check_type.is_str(testcase), False)

  def test_number_is_str_case_2(self):
    testcase = 0
    self.assertEqual(check_type.is_str(testcase), False)

  def test_number_is_str_case_3(self):
    testcase = 3.14
    self.assertEqual(check_type.is_str(testcase), False)

  def test_number_is_str_case_4(self):
    testcase = -3.14
    self.assertEqual(check_type.is_str(testcase), False)


############
# List Cases
############
# Boolean Cases
  def test_list_is_boolean_case_0(self):
    testcase = [1, 2, 3, 4]
    self.assertEqual(check_type.is_bool(testcase), False)

  def test_list_is_boolean_case_1(self):
    testcase = ["item1", "item2", "item3"]
    self.assertEqual(check_type.is_bool(testcase), False)

  def test_list_is_boolean_case_2(self):
    testcase = [True, False]
    self.assertEqual(check_type.is_bool(testcase), False)

  def test_list_is_boolean_case_2(self):
    testcase = [[1, 2, 3, 4],
                [1, 2, 3, 4]]
    self.assertEqual(check_type.is_bool(testcase), False)


# Float Cases
  def test_list_is_float_case_0(self):
    testcase = [1, 2, 3, 4]
    self.assertEqual(check_type.is_float(testcase), False)

  def test_list_is_float_case_1(self):
    testcase = ["item1", "item2", "item3"]
    self.assertEqual(check_type.is_float(testcase), False)

  def test_list_is_float_case_2(self):
    testcase = [True, False]
    self.assertEqual(check_type.is_float(testcase), False)

  def test_list_is_float_case_2(self):
    testcase = [[1, 2, 3, 4],
                [1, 2, 3, 4]]
    self.assertEqual(check_type.is_float(testcase), False)


# Integer Cases
  def test_list_is_int_case_0(self):
    testcase = [1, 2, 3, 4]
    self.assertEqual(check_type.is_int(testcase), False)

  def test_list_is_int_case_1(self):
    testcase = ["item1", "item2", "item3"]
    self.assertEqual(check_type.is_int(testcase), False)

  def test_list_is_int_case_2(self):
    testcase = [True, False]
    self.assertEqual(check_type.is_int(testcase), False)

  def test_list_is_int_case_2(self):
    testcase = [[1, 2, 3, 4],
                [1, 2, 3, 4]]
    self.assertEqual(check_type.is_int(testcase), False)


# List Cases
  def test_list_is_list_case_0(self):
    testcase = [1, 2, 3, 4]
    self.assertEqual(check_type.is_list(testcase), True)

  def test_list_is_list_case_1(self):
    testcase = ["item1", "item2", "item3"]
    self.assertEqual(check_type.is_list(testcase), True)

  def test_list_is_list_case_2(self):
    testcase = [True, False]
    self.assertEqual(check_type.is_list(testcase), True)

  def test_list_is_list_case_2(self):
    testcase = [[1, 2, 3, 4],
                [1, 2, 3, 4]]
    self.assertEqual(check_type.is_list(testcase), True)


# String Cases
  def test_list_is_str_case_0(self):
    testcase = [1, 2, 3, 4]
    self.assertEqual(check_type.is_str(testcase), False)

  def test_list_is_str_case_1(self):
    testcase = ["item1", "item2", "item3"]
    self.assertEqual(check_type.is_str(testcase), False)

  def test_list_is_str_case_2(self):
    testcase = [True, False]
    self.assertEqual(check_type.is_str(testcase), False)

  def test_list_is_str_case_2(self):
    testcase = [[1, 2, 3, 4],
                [1, 2, 3, 4]]
    self.assertEqual(check_type.is_str(testcase), False)


##############
# String Cases
##############
# Boolean Cases
  def test_string_is_boolean_case_0(self):
    testcase = "some text of string"
    self.assertEqual(check_type.is_bool(testcase), False)


# Float Cases
  def test_string_is_float_case_0(self):
    testcase = "some text of string"
    self.assertEqual(check_type.is_float(testcase), False)


# Integer Cases
  def test_string_is_int_case_0(self):
    testcase = "some text of string"
    self.assertEqual(check_type.is_int(testcase), False)


# List Cases
  def test_string_is_list_case_0(self):
    testcase = "some text of string"
    self.assertEqual(check_type.is_list(testcase), False)


# String Cases
  def test_string_is_str_case_0(self):
    testcase = "some text of string"
    self.assertEqual(check_type.is_str(testcase), True)


###############
# Special Cases
###############
# Boolean Cases
  def test_special_is_boolean_case_1(self):
    testcase = None
    self.assertEqual(check_type.is_bool(testcase), False)


# Float Cases
  def test_special_is_float_case_0(self):
    testcase = None
    self.assertEqual(check_type.is_float(testcase), False)


# Integer Cases
  def test_special_is_int_case_0(self):
    testcase = None
    self.assertEqual(check_type.is_int(testcase), False)


# List Cases
  def test_special_is_list_case_0(self):
    testcase = None
    self.assertEqual(check_type.is_list(testcase), False)


# String Cases
  def test_special_is_str_case_0(self):
    testcase = None
    self.assertEqual(check_type.is_str(testcase), False)

  if __name__ == '__main__':
    unittest.main()
