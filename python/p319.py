# %% 319. Bulb Switcher https://leetcode.com/problems/bulb-switcher/
import numpy as np


# Lessons learned:
# - Testing the array at n=50, I saw that only square numbers remained. From
#   there it was easy to prove that square numbers are the only ones with an odd
#   number of factors. So this problem is just counting the number of perfect
#   squares <= n.
def bulbSwitch(n: int) -> int:
    """
    Examples:
    >>> bulbSwitch(3)
    1
    >>> bulbSwitch(0)
    0
    >>> bulbSwitch(1)
    1
    >>> bulbSwitch(5)
    2
    """
    arr = np.zeros(n, dtype=int)
    for i in range(1, n + 1):
        for j in range(0, n):
            if (j + 1) % i == 0:
                arr[j] = 1 if arr[j] == 0 else 0
    return sum(arr)


def bulbSwitch2(n: int) -> int:
    """
    Examples:
    >>> bulbSwitch2(3)
    1
    >>> bulbSwitch2(0)
    0
    >>> bulbSwitch2(1)
    1
    >>> bulbSwitch2(5)
    2
    """
    return int(np.sqrt(n))
