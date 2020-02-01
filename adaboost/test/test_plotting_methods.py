from unittest import TestCase
from adaboost.plotting import methods

class TestPlottingMethods(TestCase):
##############
# Number Cases
##############

  # Set max xlim
  def test_number_set_max_xlim_case_0(self):
    testcase = 1
    self.assertEqual(methods.set_max_xlim(testcase), 1)

  def test_number_set_max_xlim_case_1(self):
    testcase = 9
    self.assertEqual(methods.set_max_xlim(testcase), 9)

  def test_number_set_max_xlim_case_2(self):
    testcase = 10
    self.assertEqual(methods.set_max_xlim(testcase), 10)

  def test_number_set_max_xlim_case_3(self):
    testcase = 19
    self.assertEqual(methods.set_max_xlim(testcase), 20)

  def test_number_set_max_xlim_case_4(self):
    testcase = 20
    self.assertEqual(methods.set_max_xlim(testcase), 20)

  def test_number_set_max_xlim_case_5(self):
    testcase = 40
    self.assertEqual(methods.set_max_xlim(testcase), 40)

  def test_number_set_max_xlim_case_6(self):
    testcase = 46
    self.assertEqual(methods.set_max_xlim(testcase), 50)

  def test_number_set_max_xlim_case_7(self):
    testcase = 80
    self.assertEqual(methods.set_max_xlim(testcase), 80)

  def test_number_set_max_xlim_case_8(self):
    testcase = 81
    self.assertEqual(methods.set_max_xlim(testcase), 100)

  def test_number_set_max_xlim_case_9(self):
    testcase = 150
    self.assertEqual(methods.set_max_xlim(testcase), 150)

  def test_number_set_max_xlim_case_10(self):
    testcase = 392
    self.assertEqual(methods.set_max_xlim(testcase), 400)


# Set Major x-axis
  def test_number_set_major_xaxis_case_0(self):
    testcase = 1
    self.assertEqual(methods.set_major_xaxis(testcase), 1)

  def test_number_set_major_xaxis_case_1(self):
    testcase = 9
    self.assertEqual(methods.set_major_xaxis(testcase), 1)

  def test_number_set_major_xaxis_case_2(self):
    testcase = 10
    self.assertEqual(methods.set_major_xaxis(testcase), 2)

  def test_number_set_major_xaxis_case_3(self):
    testcase = 17
    self.assertEqual(methods.set_major_xaxis(testcase), 2)

  def test_number_set_major_xaxis_case_4(self):
    testcase = 20
    self.assertEqual(methods.set_major_xaxis(testcase), 5)

  def test_number_set_major_xaxis_case_5(self):
    testcase = 36
    self.assertEqual(methods.set_major_xaxis(testcase), 5)

  def test_number_set_major_xaxis_case_6(self):
    testcase = 40
    self.assertEqual(methods.set_major_xaxis(testcase), 10)

  def test_number_set_major_xaxis_case_7(self):
    testcase = 74
    self.assertEqual(methods.set_major_xaxis(testcase), 10)

  def test_number_set_major_xaxis_case_8(self):
    testcase = 80
    self.assertEqual(methods.set_major_xaxis(testcase), 20)

  def test_number_set_major_xaxis_case_9(self):
    testcase = 120
    self.assertEqual(methods.set_major_xaxis(testcase), 20)

  def test_number_set_major_xaxis_case_10(self):
    testcase = 150
    self.assertEqual(methods.set_major_xaxis(testcase), 50)

  def test_number_set_major_xaxis_case_11(self):
    testcase = 174
    self.assertEqual(methods.set_major_xaxis(testcase), 50)


# Set Minor x-axis
  def test_number_set_minor_xaxis_case_0(self):
    testcase = 1
    self.assertEqual(methods.set_minor_xaxis(testcase), 4)

  def test_number_set_minor_xaxis_case_1(self):
    testcase = 9
    self.assertEqual(methods.set_minor_xaxis(testcase), 4)

  def test_number_set_minor_xaxis_case_2(self):
    testcase = 10
    self.assertEqual(methods.set_minor_xaxis(testcase), 4)

  def test_number_set_minor_xaxis_case_3(self):
    testcase = 17
    self.assertEqual(methods.set_minor_xaxis(testcase), 4)

  def test_number_set_minor_xaxis_case_4(self):
    testcase = 20
    self.assertEqual(methods.set_minor_xaxis(testcase), 5)

  def test_number_set_minor_xaxis_case_5(self):
    testcase = 36
    self.assertEqual(methods.set_minor_xaxis(testcase), 5)

  def test_number_set_minor_xaxis_case_6(self):
    testcase = 40
    self.assertEqual(methods.set_minor_xaxis(testcase), 5)

  def test_number_set_minor_xaxis_case_7(self):
    testcase = 74
    self.assertEqual(methods.set_minor_xaxis(testcase), 5)

  def test_number_set_minor_xaxis_case_8(self):
    testcase = 80
    self.assertEqual(methods.set_minor_xaxis(testcase), 5)

  def test_number_set_minor_xaxis_case_9(self):
    testcase = 120
    self.assertEqual(methods.set_minor_xaxis(testcase), 5)

  def test_number_set_minor_xaxis_case_10(self):
    testcase = 150
    self.assertEqual(methods.set_minor_xaxis(testcase), 5)

  def test_number_set_minor_xaxis_case_11(self):
    testcase = 174
    self.assertEqual(methods.set_minor_xaxis(testcase), 5)

  if __name__ == '__main__':
    unittest.mai1