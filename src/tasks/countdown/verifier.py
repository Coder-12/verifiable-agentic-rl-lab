from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Optional


class VerificationStatus(str, Enum):
    """Strict categorization of verification outcomes for reward mapping."""
    CORRECT = "correct"
    EMPTY_EXPRESSION = "empty_expression"
    PARSE_ERROR = "parse_error"
    UNSAFE_EXPRESSION = "unsafe_expression"
    WRONG_NUMBERS = "wrong_numbers"
    WRONG_VALUE = "wrong_value"
    DIVISION_BY_ZERO = "division_by_zero"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    is_valid: bool
    reason: VerificationStatus
    value: Optional[Fraction] = None
    used_numbers: Optional[list[int]] = None
    normalized_expression: Optional[str] = None
    error: Optional[str] = None


def _parse_expression(expression: str) -> ast.Expression:
    """
    Parses the string into an AST in eval mode.
    
    Raises:
        SyntaxError: If the string is not valid Python.
    """
    return ast.parse(expression, mode="eval")


def _validate_ast(tree: ast.AST) -> None:
    """
    Walks the AST and ensures only safe, whitelisted nodes are present.
    
    Raises:
        ValueError: If an unsafe node or non-integer constant is found.
    """
    allowed_node_types = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.UAdd,
        ast.USub,
    )
    
    for node in ast.walk(tree):
        if not isinstance(node, allowed_node_types):
            raise ValueError(f"Unsafe or unsupported AST node found: {type(node).__name__}")
    
        # Strictly enforce that constants are integers (reject floats, strings, booleans, None)
        # Note: isinstance(True, int) is True in Python, so we must use type() exact matching.
        if isinstance(node, ast.Constant):
            if type(node.value) is not int:
                raise ValueError(f"Non-integer constant found: {node.value}")


def _extract_numbers(tree: ast.AST) -> list[int]:
    """
    Collects all integer literals appearing in the expression.
    """
    numbers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            numbers.append(node.value)
    return numbers


def _evaluate_ast(node: ast.AST) -> Fraction:
    """
    Recursively evaluates the safe AST using fractions.Fraction for exact arithmetic.
    
    Raises:
        ZeroDivisionError: If division by zero occurs.
        ValueError: If an unexpected node is encountered (fallback safety).
    """
    if isinstance(node, ast.Expression):
        return _evaluate_ast(node.body)
    
    elif isinstance(node, ast.Constant):
        return Fraction(node.value, 1)
    
    elif isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.USub):
            return -operand
    
    elif isinstance(node, ast.BinOp):
        left = _evaluate_ast(node.left)
        right = _evaluate_ast(node.right)
        
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if not right:
                raise ZeroDivisionError("Divison by zero!")
            
            return left / right
    
    raise ValueError(f"Unsupported evaluation node: {type(node).__name__}")


def verify_countdown_expression(
    expression: str,
    target: int,
    numbers: list[int]
) -> VerificationResult:
    """
    Verifies a Countdown arithmetic expression against a target and a multiset of numbers.
    """
    normalized_expr = expression.strip()
    
    if not normalized_expr:
        return VerificationResult(
            is_valid=False,
            reason=VerificationStatus.EMPTY_EXPRESSION,
            normalized_expression=normalized_expr,
        )
    
    # 1. Parse
    try:
        tree = _parse_expression(expression)
    except SyntaxError as e:
        return VerificationResult(
            is_valid=False,
            reason=VerificationStatus.PARSE_ERROR,
            normalized_expression=normalized_expr,
            error=str(e),
        )
    
    # 2. Validate Safety
    try:
        _validate_ast(tree)
    except ValueError as e:
        return VerificationResult(
            is_valid=False,
            reason=VerificationStatus.UNSAFE_EXPRESSION,
            normalized_expression=normalized_expr,
            error=str(e),
        )
    
    # 3. Extract and Check Numbers (Strict Multiset Validation)
    used_numbers = _extract_numbers(tree)
    if Counter(used_numbers) != Counter(numbers):
        return VerificationResult(
            is_valid=False,
            reason=VerificationStatus.WRONG_NUMBERS,
            used_numbers=used_numbers,
            normalized_expression=normalized_expr,
            error=f"Expected {numbers}, but used {used_numbers}",
        )
    
    # 4. Evaluate Exact Value
    try:
        value = _evaluate_ast(tree)
    except ZeroDivisionError as e:
        return VerificationResult(
            is_valid=False,
            reason=VerificationStatus.DIVISION_BY_ZERO,
            used_numbers=used_numbers,
            normalized_expression=normalized_expr,
            error=str(e),
        )
    except Exception as e:
        return VerificationResult(
            is_valid=False,
            reason=VerificationStatus.UNSAFE_EXPRESSION,
            used_numbers=used_numbers,
            normalized_expression=normalized_expr,
            error=str(e),
        )
    
    # 5. Compare with Target
    if value != Fraction(target, 1):
        return VerificationResult(
            is_valid=False,
            reason=VerificationStatus.WRONG_VALUE,
            value=value,
            used_numbers=used_numbers,
            normalized_expression=normalized_expr,
            error=f"Expected {Fraction(target, 1)}, but evaluated value {value}",
        )
    

    # 6. Success
    return VerificationResult(
        is_valid=True,
        reason=VerificationStatus.CORRECT,
        value=value,
        used_numbers=used_numbers,
        normalized_expression=normalized_expr
    )
        