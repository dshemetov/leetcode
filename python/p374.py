# %% 374. Guess Number Higher or Lower https://leetcode.com/problems/guess-number-higher-or-lower/
from bisect import bisect_left

# Lessons learned:
# - bisect_left has a 'key' argument as of 3.10.
__pick__ = 6


def guess(num: int) -> int:
    if num == __pick__:
        return 0
    if num > __pick__:
        return -1
    return 1


def guessNumber(n: int) -> int:
    """
    Examples:
    >>> guessNumber(10)
    6
    """
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        out = guess(mid)
        if out == 1:
            lo = mid + 1
        elif out == -1:
            hi = mid - 1
        else:
            return mid

    return lo


def guessNumber2(n: int) -> int:
    """
    Examples:
    >>> guessNumber2(10)
    6
    """
    return bisect_left(range(0, n), 0, lo=0, hi=n, key=lambda x: -guess(x))
