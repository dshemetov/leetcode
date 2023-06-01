# %% 316. Remove Duplicate Letters https://leetcode.com/problems/remove-duplicate-letters/
# 1081. https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/
from collections import Counter


# Lessons learned:
# - In this one, the solution heuristic can be established with a few examples.
#   The key is that we can greedily remove left-most duplicated letters that are
#   larger than the next letter. For example, if we have cbxxx and we can remove
#   c or another letter, then we will have bxxx < cbxx.
def removeDuplicateLetters(s: str) -> str:
    """
    Examples:
    >>> removeDuplicateLetters("bcabc")
    'abc'
    >>> removeDuplicateLetters("cbacdcbc")
    'acdb'
    >>> removeDuplicateLetters("bbcaac")
    'bac'
    >>> removeDuplicateLetters("bcba")
    'bca'
    """
    letter_counts = Counter(s)
    stack = []
    for c in s:
        letter_counts[c] -= 1
        if c in stack:
            continue
        while stack and c < stack[-1] and letter_counts[stack[-1]] > 0:
            stack.pop()
        stack.append(c)
    return "".join(stack)
