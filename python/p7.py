# %% 7. Reverse Integer https://leetcode.com/problems/reverse-integer/


# Lessons learned:
# - The most interesting part of this problem is finding out how to check for
#   overflow without overflowing. This can be done by checking whether the
#   multiplication by 10 will overflow or if the the multiplication by 10 will
#   bring you right to the edge and the next digit will overflow.
# - Another interesting part is that Python's modulo operator behaves
#   differently than in C. The modulo operator performs Euclidean division a = b
#   * q + r, where r is the remainder and q is the quotient. In Python, the
#   remainder r is always positive, whereas in C, the remainder r has the same
#   sign as the dividend a. This in turn, implies that in C, q = truncate(a/b),
#   while in Python, q = floor(a/b). So in Python, -(-x % n) = -((n - x % n) %
#   n), while in C, we have (-x % n) = -(x % n). Also, in Python, -(-x // n) =
#   (x - 1) // n + 1.
def reverse(x: int) -> int:
    """
    Examples:
    >>> reverse(123)
    321
    >>> reverse(-123)
    -321
    >>> reverse(120)
    21
    >>> reverse(-1563847412)
    0
    >>> reverse(-10)
    -1
    """
    int_max_div10 = (2**31 - 1) // 10
    int_min_div10 = (-(2**31)) // 10 + 1

    rx = 0
    while x != 0:
        if x < 0:
            r = -((10 - x % 10) % 10)
            x = (x - 1) // 10 + 1
        else:
            r = x % 10
            x //= 10

        if rx > int_max_div10 or (rx == int_max_div10 and r > 7):
            return 0
        if rx < int_min_div10 or (rx == int_min_div10 and r < -8):
            return 0

        rx = rx * 10 + r

    return rx
