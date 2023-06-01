# %% 12. Integer to Roman https://leetcode.com/problems/integer-to-roman/
def int_to_roman(num: int) -> str:
    """
    Examples:
    >>> int_to_roman(3)
    'III'
    >>> int_to_roman(4)
    'IV'
    >>> int_to_roman(9)
    'IX'
    >>> int_to_roman(58)
    'LVIII'
    >>> int_to_roman(1994)
    'MCMXCIV'
    """
    letter_map = {
        1: "I",
        4: "IV",
        5: "V",
        9: "IX",
        10: "X",
        40: "XL",
        50: "L",
        90: "XC",
        100: "C",
        400: "CD",
        500: "D",
        900: "CM",
        1000: "M",
    }
    s = ""
    for k in sorted(letter_map.keys(), reverse=True):
        r, num = divmod(num, k)
        s += letter_map[k] * r

    return s
