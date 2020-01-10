from adaboost.common.check import check_type

# Extract value from argument keyword parameter
def arg_value(keyword, raw_input):
  # Ensure both parameters are of type string
  type_check = (
    not check_type.is_str(keyword)
    or not check_type.is_str(raw_input)
  )
  if type_check:
    return None

  # Remove only the first leading instance, whilst strip any preceding/ending spaces
  value = raw_input.replace(keyword, '', 1).strip()
  if value is '':
    return None

  # Ensure preceding character is of an assignment type
  if value[0] == '=':
    value = value.replace('=', '', 1).strip()
    if value is not '':
      return value

  return None
