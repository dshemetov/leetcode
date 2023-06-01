# %% 9. Palindrome Number https://leetcode.com/problems/palindrome-number/
import math


def is_palindrome(x: int) -> bool:
    """
    Examples:
    >>> is_palindrome(121)
    True
    >>> is_palindrome(-121)
    False
    >>> is_palindrome(10)
    False
    >>> is_palindrome(0)
    True
    >>> is_palindrome(1)
    True
    >>> is_palindrome(1331)
    True
    >>> is_palindrome(1332)
    False
    >>> is_palindrome(133454331)
    True
    >>> is_palindrome(1122)
    False
    """
    if x < 0:
        return False
    if x == 0:
        return True

    num_digits = math.floor(math.log10(x)) + 1

    if num_digits == 1:
        return True

    for i in range(num_digits // 2):
        if x // 10**i % 10 != x // 10 ** (num_digits - i - 1) % 10:
            return False
    return True
