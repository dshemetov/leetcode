# %% 8. String to Integer (atoi) https://leetcode.com/problems/string-to-integer-atoi/
def my_atoi(s: str) -> int:
    """
    Examples:
    >>> my_atoi("42")
    42
    >>> my_atoi("   -42")
    -42
    >>> my_atoi("4193 with words")
    4193
    >>> my_atoi("words and 987")
    0
    >>> my_atoi("-91283472332")
    -2147483648
    >>> my_atoi("91283472332")
    2147483647
    >>> my_atoi("3.14159")
    3
    >>> my_atoi("+-2")
    0
    >>> my_atoi("  -0012a42")
    -12
    >>> my_atoi("  +0 123")
    0
    >>> my_atoi("-0")
    0
    """
    _max = 2**31 - 1
    _min = 2**31

    factor = 0
    num = 0
    digits_started = False
    for c in s:
        if not digits_started:
            if c == " ":
                continue
            if c == "-":
                factor = -1
                digits_started = True
            elif c == "+":
                factor = 1
                digits_started = True
            elif c.isdigit():
                factor = 1
                digits_started = True
                num = int(c)
            else:
                return 0
        else:
            if c.isdigit():
                num = num * 10 + int(c)
                if factor == 1 and num > _max:
                    return _max
                if factor == -1 and num > _min:
                    return -_min
            else:
                break

    return factor * num
