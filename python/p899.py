# %% 899. Orderly Queue https://leetcode.com/problems/orderly-queue/


# Lessons learned:
# - This problem is such a troll. At first I thought I found a totally
#   ridiculous Copilot suggestion, but then I realized that the solution was
#   actually dead simple - you can use the rightmost character as a register and
#   rotate the string until the correct insertion point.
def orderlyQueue(s: str, k: int) -> str:
    """
    Examples:
    >>> orderlyQueue("cba", 1)
    'acb'
    >>> orderlyQueue("baaca", 3)
    'aaabc'
    >>> orderlyQueue("baaca", 1)
    'aacab'
    >>> orderlyQueue("baaca", 2)
    'aaabc'
    >>> orderlyQueue("baaca", 4)
    'aaabc'
    >>> orderlyQueue("badaca", 2)
    'aaabcd'
    >>> orderlyQueue("badacadeff", 3)
    'aaabcddeff'
    """
    if k == 1:
        return min(s[i:] + s[:i] for i in range(len(s)))

    return "".join(sorted(s))
