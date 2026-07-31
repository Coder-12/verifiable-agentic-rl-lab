from src.tasks.countdown.prompts import (
    COUNTDOWN_SYSTEM_PROMPT,
    extract_answer,
    format_countdown_prompt,
)

def test_format_countdown_prompt():
    numbers = [1, 2, 3, 4]
    target = 24
    
    prompt = format_countdown_prompt(numbers, target)
    
    assert "Numbers: [1, 2, 3, 4]" in prompt
    assert "Target: 24" in prompt
    assert "Use exactly these numbers once each: [1, 2, 3, 4]" in prompt
    assert "Do not introduce any other integer literals." in prompt


def test_system_prompt_warns_against_extra_integer_literals():
    assert "Do not introduce any other integer literal" in COUNTDOWN_SYSTEM_PROMPT
    assert "Do not use decimal literals in the final answer." in COUNTDOWN_SYSTEM_PROMPT
    assert "every given number appears exactly once" in COUNTDOWN_SYSTEM_PROMPT
    
def test_extract_answer_valid():
    completion = """<reasoning>
    I need to make 24.
    6 / (1 - 3/4) = 24.
    </reasoning>
    <answer>6 / (1 - 3/4)</answer>
    """
    extracted = extract_answer(completion)
    
    assert extracted == "6 / (1 - 3/4)"

def test_extract_answer_with_newlines_and_spaces():
    completion = """<reasoning>Thinking...</reasoning>
    <answer>
        6 / (1 - 3/4)
    </answer>
    """
    extracted = extract_answer(completion)
    
    assert extracted == "6 / (1 - 3/4)"

def test_extract_answer_with_missing_tag():
    completion = """<reasoning>Thinking...</reasoning>
    The answer is 6 / (1 - 3/4)
    """
    extracted = extract_answer(completion)
    
    assert extracted is None

def test_extract_answer_multiple_tags_fails():
    # We strictly enforce exactly ONE answer block to prevent reward hacking
    completion = """<answer>1 + 3</answer> wait no, <answer>6 / (1 - 3/4)</answer>"""
    
    extracted = extract_answer(completion)
    assert extracted is None

def test_extract_answer_malformed_tags():
    completion = "<answer> 24 </wrong_close>"
    
    extracted = extract_answer(completion)
    assert extracted is None
