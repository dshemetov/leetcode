# %% 1323. Maximum 69 Number https://leetcode.com/problems/maximum-69-number/
import math


# Lessons learned:
# - Converting to a string and using replace is surprisingly fast.
# - Just need to accept that Python string built-ins are in C-land.
def maximum69Number(num: int) -> int:
    """
    Examples:
    >>> maximum69Number(9669)
    9969
    >>> maximum69Number(9996)
    9999
    >>> maximum69Number(9999)
    9999
    """
    for i in range(math.floor(math.log10(num)) + 1, -1, -1):
        if num // 10**i % 10 == 6:
            return num + 3 * 10**i
    return num


def maximum69Number2(num: int) -> int:
    return int(str(num).replace("6", "9", 1))
