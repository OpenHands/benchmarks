import re
import string
import warnings
from typing import Any


def normalize_number_str(number_str: str) -> float:
    """Normalize a number string by removing common units and commas."""
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        print(f"String {number_str} cannot be normalized to number str.")
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] | None = None,
) -> list[str]:
    """Split a string by a list of characters."""
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase

    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)

    Returns:
    - str, the normalized string
    """
    no_spaces = re.sub(r"\s", "", input_str)

    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


def question_scorer(
    model_answer: str,
    ground_truth: str,
) -> bool:
    """Score a model answer against ground truth.

    Handles three types of answers:
    1. Numbers (with unit normalization)
    2. Comma/semicolon separated lists
    3. Strings (with normalization)
    """

    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    # If ground truth is a number
    if is_float(ground_truth):
        print(f"Evaluating {model_answer} as a number.")
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # If ground truth is a list
    elif any(char in ground_truth for char in [",", ";"]):
        print(f"Evaluating {model_answer} as a comma separated list.")

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        # Check length is the same
        if len(gt_elems) != len(ma_elems):
            warnings.warn(
                "Answer lists have different lengths, returning False.",
                UserWarning,
                stacklevel=2,
            )
            return False

        # Compare each element as float or str
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # If ground truth is a string
    else:
        print(f"Evaluating {model_answer} as a string.")
        return normalize_str(model_answer) == normalize_str(ground_truth)
