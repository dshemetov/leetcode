# %% 17. Letter Combinations of a Phone Number https://leetcode.com/problems/letter-combinations-of-a-phone-number/
from itertools import product


# Lessons learned:
# - Implement the Cartesian product, they say! It's fun, they say! And it turns
#   out, that it is kinda fun.
# - The second solution is a little more manual, but worth knowing.
def letter_combinations(digits: str) -> list[str]:
    """
    Examples:
    >>> letter_combinations("23")
    ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
    >>> letter_combinations("")
    []
    >>> letter_combinations("2")
    ['a', 'b', 'c']
    """
    if not digits:
        return []

    letter_map = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }

    res = product(*[letter_map[c] for c in digits])
    return ["".join(t) for t in res]


def letter_combinations2(digits: str) -> list[str]:
    """
    Examples:
    >>> letter_combinations("23")
    ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
    >>> letter_combinations("")
    []
    >>> letter_combinations("2")
    ['a', 'b', 'c']
    """
    if not digits:
        return []

    letter_map = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }

    res = []
    for c in digits:
        if not res:
            res = list(letter_map[c])
        else:
            res = [a + b for a in res for b in letter_map[c]]

    return res
