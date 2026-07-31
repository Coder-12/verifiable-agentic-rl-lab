import os
import sys
from fractions import Fraction

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
    
from src.tasks.countdown.verifier import verify_countdown_expression, VerificationStatus


def test_valid_expression_accepted():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "6 / (1 - 3/4)"
    
    result = verify_countdown_expression(expression, target, numbers)
    assert result.is_valid is True
    assert result.reason == VerificationStatus.CORRECT
    assert result.value == Fraction(24, 1)
    # Order doesn't matter for the assert if we sort, but our extractor walks left-to-right based on how it processed the expression
    assert sorted(result.used_numbers) == sorted(numbers)

def test_wrong_value_rejected():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "1 + 3 + 4 + 6"
    
    result = verify_countdown_expression(expression, target, numbers)
    
    assert result.is_valid is False
    assert result.reason == VerificationStatus.WRONG_VALUE
    assert result.value == Fraction(14, 1)

def test_reused_number_rejected():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "6 / (1 - 3/3)"  # Uses 3 twice, misses 4
    
    result = verify_countdown_expression(expression, target, numbers)
    
    assert result.is_valid is False
    assert result.reason == VerificationStatus.WRONG_NUMBERS


def test_missing_number_rejected():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "6 / (1 - 3)"  # Misses 4
    
    result = verify_countdown_expression(expression, target, numbers)
    
    assert result.is_valid is False
    assert result.reason == VerificationStatus.WRONG_NUMBERS
    
def test_extra_number_rejected():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "6 / (1 - 3/4) + 9"  # Uses 9
    
    result = verify_countdown_expression(expression, target, numbers)
    
    assert result.is_valid is False
    assert result.reason == VerificationStatus.WRONG_NUMBERS
    
def test_unsafe_expression_rejected():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "__import__('os').system('echo hacked')"
    
    result = verify_countdown_expression(expression, target, numbers)
    
    assert result.is_valid is False
    assert result.reason == VerificationStatus.UNSAFE_EXPRESSION

def test_variable_name_rejected():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "a + 1 + 3 + 4"
    
    result = verify_countdown_expression(expression, target, numbers)
    
    assert result.is_valid is False
    assert result.reason == VerificationStatus.UNSAFE_EXPRESSION
    

def test_function_call_rejected():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "abs(1 - 3) + 4 + 6"
    
    result = verify_countdown_expression(expression, target, numbers)
    
    assert result.is_valid is False
    assert result.reason == VerificationStatus.UNSAFE_EXPRESSION

def test_division_by_zero_rejected():
    # We use a specific multiset to ensure WRONG_NUMBERS doesn't trigger first
    numbers = [1, 1, 2, 3]
    target = 24
    expression = "2 / (1 - 1) + 3"
    
    result = verify_countdown_expression(expression, target, numbers)
    
    assert result.is_valid is False
    assert result.reason == VerificationStatus.DIVISION_BY_ZERO

def test_duplicate_numbers_handled_correctly():
    numbers = [2, 2, 5, 10]
    target = 20
    expression = "(10 - 5) * (2 + 2)"
    
    result = verify_countdown_expression(expression, target, numbers)
    
    assert result.is_valid is True
    assert result.reason == VerificationStatus.CORRECT
    assert result.value == Fraction(20, 1)

def test_empty_expression_rejected():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "   "
    
    result = verify_countdown_expression(expression, target, numbers)
    
    assert result.is_valid is False
    assert result.reason == VerificationStatus.EMPTY_EXPRESSION

def test_float_and_bool_rejected():
    # Ensures our strict `type(node.value) is int` check works
    numbers = [1, 3, 4, 6]
    target = 24
    
    result_float = verify_countdown_expression("6.0 / (1 - 3/4)", target, numbers)
    assert result_float.is_valid is False
    assert result_float.reason == VerificationStatus.UNSAFE_EXPRESSION
    
    result_bool = verify_countdown_expression("6 / (True - 3/4)", target, numbers)
    assert result_bool.is_valid is False
    assert result_bool.reason == VerificationStatus.UNSAFE_EXPRESSION
