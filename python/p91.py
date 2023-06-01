# %% 91. Decode Ways https://leetcode.com/problems/decode-ways/


# Lessons learned:
# - My first observation was that characters in "34567890" acted as separators,
#   where I could split the string into substrings that could be decoded
#   independently. My second observation was that the number of ways to decode a
#   string of nothing but "1" and "2" characters was the Fibonacci number
#   F(n+1), where n is the number of characters in the string. Combining these
#   two speedups with recursion gave me the first solution, which had middle of
#   the pack runtime and memory usage.
# - The second solution is a very clean dynamic programming approach I lifted
#   from the discussion section. Define
#
#       dp(i) = number of ways to decode the substring s[:i]
#
#   The recurrence relation is
#
#       dp(i) = dp(i-1) + dp(i-2) if "11" <= s[i-2:i] <= "26" and s[i-2:i] != "20"
#             = dp(i-1) if s[i-1] != "0" and s[i-2:i] > "26"
#             = dp(i-2) if "10" == s[i-2:i] or "20" == s[i-2:i]
#             = 0 otherwise
#       dp(0) = 1
#       dp(1) = 1 if s[0] != "0" else 0
#
# - Fun fact: the number of binary strings of length n with no consecutive zeros
#   corresponds to the Fibonacci number F(n+2). This diagram helps visualize the
#   recursion:
#   https://en.wikipedia.org/wiki/Composition_(combinatorics)#/media/File:Fibonacci_climbing_stairs.svg.
def get_fibonacci_number(n: int) -> int:
    """Return the nth Fibonacci number, where F(0) = 0 and F(1) = 1.

    Examples:
    >>> [get_fibonacci_number(i) for i in range(10)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
    phi = (1 + 5**0.5) / 2
    return int(phi**n / 5**0.5 + 0.5)


def get_binary_strings_no_consecutive_zeros(n: int) -> list[str]:
    """
    Examples:
    >>> get_binary_strings_no_consecutive_zeros(1)
    ['0', '1']
    >>> get_binary_strings_no_consecutive_zeros(2)
    ['01', '11', '10']
    >>> get_binary_strings_no_consecutive_zeros(3)
    ['011', '111', '101', '010', '110']
    >>> [len(get_binary_strings_no_consecutive_zeros(i)) for i in range(1, 10)]
    [2, 3, 5, 8, 13, 21, 34, 55, 89]
    """
    if n == 1:
        return ["0", "1"]

    prev = get_binary_strings_no_consecutive_zeros(n - 1)
    return [x + "1" for x in prev] + [x + "0" for x in prev if not x.endswith("0")]


def numDecodings(s: str) -> int:
    """
    Examples:
    >>> numDecodings("0")
    0
    >>> numDecodings("06")
    0
    >>> numDecodings("1")
    1
    >>> numDecodings("12")
    2
    >>> numDecodings("111")
    3
    >>> numDecodings("35")
    1
    >>> numDecodings("226")
    3
    >>> numDecodings("2020")
    1
    >>> numDecodings("2021")
    2
    >>> numDecodings("2022322")
    6
    """
    valid_codes = {str(x) for x in range(1, 27)}

    def recurse(s1: str) -> int:
        if s1[0] == "0":
            return 0

        if len(s1) == 1:
            return 1

        if len(s1) == 2:
            if int(s1) <= 26 and s1[1] != "0":
                return 2
            if int(s1) <= 26 and s1[1] == "0":
                return 1
            if int(s1) > 26 and s1[1] != "0":
                return 1
            return 0

        if set(s1) <= {"1", "2"}:
            return get_fibonacci_number(len(s1) + 1)

        if s1[:2] in valid_codes:
            return recurse(s1[1:]) + recurse(s1[2:])

        return recurse(s1[1:])

    total = 1
    prev = 0
    for i, c in enumerate(s):
        if c in "34567890":
            total *= recurse(s[prev : i + 1])
            prev = i + 1

    if s[prev:]:
        total *= recurse(s[prev:])

    return total


def numDecodings2(s: str) -> int:
    """
    Examples:
    >>> numDecodings2("0")
    0
    >>> numDecodings2("06")
    0
    >>> numDecodings2("1")
    1
    >>> numDecodings2("12")
    2
    >>> numDecodings2("111")
    3
    >>> numDecodings2("35")
    1
    >>> numDecodings2("226")
    3
    >>> numDecodings2("2020")
    1
    >>> numDecodings2("2021")
    2
    >>> numDecodings2("2022322")
    6
    """
    if not s or s[0] == "0":
        return 0

    dp = [0] * (len(s) + 1)
    dp[0:2] = [1, 1]

    for i in range(2, len(s) + 1):
        if "11" <= s[i - 2 : i] <= "19" or "21" <= s[i - 2 : i] <= "26":
            dp[i] = dp[i - 1] + dp[i - 2]
        elif s[i - 1] != "0":
            dp[i] = dp[i - 1]
        elif "10" == s[i - 2 : i] or "20" == s[i - 2 : i]:
            dp[i] = dp[i - 2]

    return dp[-1]
