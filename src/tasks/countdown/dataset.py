import random
from fractions import Fraction
from typing import TypedDict


class CountdownTask(TypedDict):
    """The exact dictionary structure yielded by our dataset generator."""
    numbers: list[int]
    target: int
    expected_solution: str


def _generate_random_expression(numbers: list[int]) -> tuple[str, Fraction]:
    """
    Takes a list of numbers and randomly combines them into a valid mathematical 
    expression string and its exact Fraction value.
    """
    # Start with each number as a separate node: (expression_string, evaluated_value)
    nodes = [(str(n), Fraction(n, 1)) for n in numbers]
    
    # Randomize the order we combine them
    random.shuffle(nodes)
    
    while len(nodes) > 1:
        # Pop two nodes to combine
        right_str, right_val = nodes.pop()
        left_str, left_val = nodes.pop()
        
        # Shuffle operators to ensure variety
        ops = ['+', '-', '*', '/']
        random.shuffle(ops)
        
        for op in ops:
            # Prevent division by zero
            if op == '/' and right_val == 0:
                continue
                
            if op == '+':
                new_val = left_val + right_val
            elif op == '-':
                new_val = left_val - right_val
            elif op == '*':
                new_val = left_val * right_val
            elif op == '/':
                new_val = left_val / right_val
                
            # Wrap in parentheses to preserve order of operations
            new_str = f"({left_str} {op} {right_str})"
            nodes.append((new_str, new_val))
            break  # Successfully combined, move to the next iteration
            
    return nodes[0]


def generate_countdown_dataset(
    num_samples: int,
    min_numbers: int = 3,
    max_numbers: int = 5,
    min_value: int = 1,
    max_value: int = 20
) -> list[CountdownTask]:
    """
    Generates a dataset of guaranteed-solvable Countdown tasks.
    
    Args:
        num_samples: How many unique tasks to generate.
        min_numbers: Minimum amount of numbers in the puzzle.
        max_numbers: Maximum amount of numbers in the puzzle.
        min_value: Minimum integer value for the input numbers.
        max_value: Maximum integer value for the input numbers.
        
    Returns:
        A list of CountdownTask dictionaries.
    """
    dataset: list[CountdownTask] = []
    
    while len(dataset) < num_samples:
        # 1. Pick a random amount of numbers for this puzzle
        k = random.randint(min_numbers, max_numbers)
        
        # 2. Sample random integers
        numbers = [random.randint(min_value, max_value) for _ in range(k)]
        
        # 3. Build a random expression
        expr_str, expr_val = _generate_random_expression(numbers.copy())
        
        # 4. Strict constraints: Target must be a positive integer
        if expr_val.denominator == 1 and expr_val.numerator > 0:
            target = int(expr_val.numerator)
            
            dataset.append({
                "numbers": sorted(numbers),  # Sort for clean presentation
                "target": target,
                "expected_solution": expr_str
            })
            
    return dataset
