# %% 1680. Concatenation of Consecutive Binary Numbers https://leetcode.com/problems/concatenation-of-consecutive-binary-numbers/
import math


def concatenatedBinary(n: int) -> int:
    """
    Examples:
    >>> concatenatedBinary(1)
    1
    >>> concatenatedBinary(3)
    27
    >>> concatenatedBinary(12)
    505379714
    """
    M = 10**9 + 7
    total = 1
    for i in range(2, n + 1):
        total = ((total << math.floor(math.log2(i)) + 1) + i) % M

    return total
