# %% 258. Add Digits https://leetcode.com/problems/add-digits/


# Lessons learned:
# - Turns out this can be solved with modular arithmetic because 10 ** n == 1 mod 9
def addDigits(num: int) -> int:
    """
    Examples:
    >>> addDigits(38)
    2
    >>> addDigits(0)
    0
    """
    if num == 0:
        return num
    if num % 9 == 0:
        return 9
    return num % 9
