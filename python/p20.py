from collections import deque


# %% 20. Valid Parentheses https://leetcode.com/problems/valid-parentheses/
def isValid(s: str) -> bool:
    """
    Examples:
    >>> isValid("()")
    True
    >>> isValid("()[]{}")
    True
    >>> isValid("(]")
    False
    """
    stack = deque()
    bracket_map = {"(": ")", "[": "]", "{": "}"}
    for c in s:
        if c in bracket_map:
            stack.append(c)
        elif not stack or bracket_map[stack.pop()] != c:
            return False

    return not stack
