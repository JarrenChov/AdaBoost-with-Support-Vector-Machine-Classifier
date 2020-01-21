from unittest import skip, TestCase
from adaboost.common import convert_type

# Notes:
# =============================================================================
#   Some cases may have a skip, this is intentional as currently I personally
#   am not sure how to deal with such - Primarily floating nan, inf.
#
# The following cases and its identifier detail the information corresponding to its information.
# UNDEF_CASE_NAN - A case where nan will be ignored.
# UNDEF_CASE_INF - A case where inf will be ignored.
# =============================================================================

class TestConvertType(TestCase):

###############
# Boolean Cases
###############
# Boolean Cases
  def test_boolean_boolean_case_0(self):
    testcase = True
    self.assertEqual(convert_type.to_bool(testcase), True)

  def test_boolean_boolean_case_1(self):
    testcase = False
    self.assertEqual(convert_type.to_bool(testcase), False)

# Float Cases
  def test_boolean_float_case_0(self):
    testcase = True
    self.assertEqual(convert_type.to_float(testcase), None)

  def test_boolean_float_case_1(self):
    testcase = False
    self.assertEqual(convert_type.to_float(testcase), None)

# Integer Cases
  def test_boolean_int_case_0(self):
    testcase = True
    self.assertEqual(convert_type.to_int(testcase), None)

  def test_boolean_int_case_1(self):
    testcase = False
    self.assertEqual(convert_type.to_int(testcase), None)

# String Cases
  def test_boolean_string_case_0(self):
    testcase = True
    self.assertEqual(convert_type.to_string(testcase), 'True')

  def test_boolean_string_case_1(self):
    testcase = False
    self.assertEqual(convert_type.to_string(testcase), 'False')


############
# List Cases
############
# Boolean Cases
  def test_list_boolean_case_0(self):
    testcase = [0, 1, 2, 3]
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_list_boolean_case_1(self):
    testcase = [[0], [1], [2]]
    self.assertEqual(convert_type.to_bool(testcase), None)

# Float Cases
  def test_list_float_case_0(self):
    testcase = [0, 1, 2, 3]
    self.assertEqual(convert_type.to_float(testcase), None)

  def test_list_float_case_1(self):
    testcase = [[0], [1], [2]]
    self.assertEqual(convert_type.to_float(testcase), None)

# Integer Cases
  def test_list_int_case_0(self):
    testcase = [0, 1, 2, 3]
    self.assertEqual(convert_type.to_int(testcase), None)

  def test_list_int_case_1(self):
    testcase = [[0], [1], [2]]
    self.assertEqual(convert_type.to_int(testcase), None)

# String Cases
  def test_list_string_case_0(self):
    testcase = [0, 1, 2, 3]
    self.assertEqual(convert_type.to_string(testcase), None)

  def test_list_string_case_0(self):
    testcase = [[0], [1], [2]]
    self.assertEqual(convert_type.to_string(testcase), None)


############
# Null Cases
############
# Boolean Cases
  def test_nulls_bool_case_0(self):
    testcase = float('nan')
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_null_bool_case_1(self):
    testcase = None
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_null_bool_case_2(self):
    testcase = float('inf')
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_null_bool_case_3(self):
    testcase = -float('inf')
    self.assertEqual(convert_type.to_bool(testcase), None)

# Float Cases
  @skip("UNDEF_CASE_NAN")
  def test_nulls_float_case_0(self):
    testcase = float('nan')
    self.assertEqual(convert_type.to_float(testcase), None)

  def test_null_float_case_1(self):
    testcase = None
    self.assertEqual(convert_type.to_float(testcase), None)

  @skip("UNDEF_CASE_INF")
  def test_null_float_case_2(self):
    testcase = float('inf')
    self.assertEqual(convert_type.to_float(testcase), None)

  @skip("UNDEF_CASE_INF")
  def test_null_float_case_3(self):
    testcase = -float('inf')
    self.assertEqual(convert_type.to_float(testcase), None)

# Integer Cases
  def test_nulls_int_case_0(self):
    testcase = float('nan')
    self.assertEqual(convert_type.to_int(testcase), None)

  def test_null_int_case_1(self):
    testcase = None
    self.assertEqual(convert_type.to_int(testcase), None)

  def test_null_int_case_2(self):
    testcase = float('inf')
    self.assertEqual(convert_type.to_int(testcase), None)

  def test_null_int_case_3(self):
    testcase = -float('inf')
    self.assertEqual(convert_type.to_int(testcase), None)

# String Cases
  def test_null_string_case_0(self):
    testcase = float('nan')
    self.assertEqual(convert_type.to_string(testcase), 'nan')

  def test_null_string_case_1(self):
    testcase = None
    self.assertEqual(convert_type.to_string(testcase), None)

  def test_null_string_case_2(self):
    testcase = float('inf')
    self.assertEqual(convert_type.to_string(testcase), 'inf')

  def test_null_string_case_3(self):
    testcase = -float('inf')
    self.assertEqual(convert_type.to_string(testcase), '-inf')


######################
# Number / Float Cases
######################
# Boolean Cases
  def test_number_bool_case_0(self):
    testcase = -1
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_number_bool_case_1(self):
    testcase = 1048
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_number_bool_case_2(self):
    testcase = 3.14
    self.assertEqual(convert_type.to_bool(testcase), None)

# Float Cases
  def test_number_float_case_0(self):
    testcase = -1
    self.assertEqual(convert_type.to_float(testcase), -1.0)

  def test_number_float_case_1(self):
    testcase = 1048
    self.assertEqual(convert_type.to_float(testcase), 1048.0)

  def test_number_float_case_2(self):
    testcase = 3.14
    self.assertEqual(convert_type.to_float(testcase), 3.14)

# Integer Cases
  def test_number_int_case_0(self):
    testcase = -1
    self.assertEqual(convert_type.to_int(testcase), -1)

  def test_number_int_case_1(self):
    testcase = 1048
    self.assertEqual(convert_type.to_int(testcase), 1048)

  def test_number_int_case_2(self):
    testcase = 3.14
    self.assertEqual(convert_type.to_int(testcase), None)

# String Cases
  def test_number_string_case_0(self):
    testcase = -1
    self.assertEqual(convert_type.to_string(testcase), '-1')

  def test_number_string_case_1(self):
    testcase = 1048
    self.assertEqual(convert_type.to_string(testcase), '1048')

  def test_number_string_case_2(self):
    testcase = 3.14
    self.assertEqual(convert_type.to_string(testcase), '3.14')


##############
# String Cases
##############
# Boolean Cases
  def test_string_bool_case_0(self):
    testcase = "1392"
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_string_bool_case_1(self):
    testcase = "A mEsSeY sTrInG"
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_string_bool_case_2(self):
    testcase = "452-34"
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_string_bool_case_3(self):
    testcase = "!@!-9*)"
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_string_bool_case_4(self):
    testcase = "true"
    self.assertEqual(convert_type.to_bool(testcase), True)

  def test_string_bool_case_5(self):
    testcase = "false"
    self.assertEqual(convert_type.to_bool(testcase), False)

  def test_string_bool_case_6(self):
    testcase = "nTrue"
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_string_bool_case_7(self):
    testcase = "nFalse"
    self.assertEqual(convert_type.to_bool(testcase), None)

  def test_string_bool_case_8(self):
    testcase = "FALSE"
    self.assertEqual(convert_type.to_bool(testcase), False)

  def test_string_bool_case_9(self):
    testcase = "TRUE"
    self.assertEqual(convert_type.to_bool(testcase), True)

  def test_string_bool_case_10(self):
    testcase = "fAlSe"
    self.assertEqual(convert_type.to_bool(testcase), False)

  def test_string_bool_case_11(self):
    testcase = "tRuE"
    self.assertEqual(convert_type.to_bool(testcase), True)

# Float Cases
  def test_string_float_case_0(self):
    testcase = "1392"
    self.assertEqual(convert_type.to_float(testcase), 1392)

  def test_string_float_case_1(self):
    testcase = "A mEsSeY sTrInG"
    self.assertEqual(convert_type.to_float(testcase), None)

  def test_string_float_case_2(self):
    testcase = "452-34"
    self.assertEqual(convert_type.to_float(testcase), None)

  def test_string_float_case_3(self):
    testcase = "!@!-9*)"
    self.assertEqual(convert_type.to_float(testcase), None)

# Integer Cases
  def test_string_int_case_0(self):
    testcase = "1392"
    self.assertEqual(convert_type.to_int(testcase), 1392)

  def test_string_int_case_1(self):
    testcase = "A mEsSeY sTrInG"
    self.assertEqual(convert_type.to_int(testcase), None)

  def test_string_int_case_2(self):
    testcase = "452-34"
    self.assertEqual(convert_type.to_int(testcase), None)

  def test_string_int_case_3(self):
    testcase = "!@!-9*)"
    self.assertEqual(convert_type.to_int(testcase), None)

# String Cases
  def test_string_string_case_0(self):
    testcase = "1392"
    self.assertEqual(convert_type.to_string(testcase), '1392')

  def test_string_string_case_1(self):
    testcase = "A mEsSeY sTrInG"
    self.assertEqual(convert_type.to_string(testcase), 'A mEsSeY sTrInG')

  def test_string_string_case_2(self):
    testcase = "452-34"
    self.assertEqual(convert_type.to_string(testcase), '452-34')

  def test_string_string_case_3(self):
    testcase = "!@!-9*)"
    self.assertEqual(convert_type.to_string(testcase), '!@!-9*)')

if __name__ == '__main__':
  unittest.main()
