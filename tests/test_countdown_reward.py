from src.tasks.countdown.reward import countdown_reward

def test_correct_expression_gives_full_reward():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "6 / (1 - 3/4)"
    
    reward = countdown_reward(expression, target, numbers)
    assert reward == 1.0

def test_wrong_value_gives_partial_reward():
    numbers = [1, 3, 4, 6]
    target = 24
    # Valid math, uses correct numbers, but evaluates to 14
    expression = "1 + 3 + 4 + 6"
    
    reward = countdown_reward(expression, target, numbers)
    assert reward == 0.3


def test_wrong_numbers_gives_partial_reward():
    numbers = [1, 3, 4, 6]
    target = 24
    # Valid math, but misses the 4
    expression = "6 / (1 - 3)"
    
    reward = countdown_reward(expression, target, numbers)
    assert reward == 0.05


def test_unsafe_expression_gives_zero_reward():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "__import__('os').system('echo hacked')"
    
    reward = countdown_reward(expression, target, numbers)
    assert reward == 0.0
    
def test_empty_expression_gives_zero_reward():
    numbers = [1, 3, 4, 6]
    target = 24
    expression = "   "
    
    reward = countdown_reward(expression, target, numbers)
    assert reward == 0.0


def test_division_by_zero_gives_zero_reward():
    numbers = [1, 1, 2, 3]
    target = 24
    expression = "2 / (1 - 1) + 3"
    
    reward = countdown_reward(expression, target, numbers)
    assert reward == 0.0
