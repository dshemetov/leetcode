# %% 5. Longest Palindromic Substring https://leetcode.com/problems/longest-palindromic-substring/


# Lessons learned:
# - I tried an approach with three pointers and expanding outwards if the
#   characters matched. The edge case that stumped me was handling long runs of
#   the same character such as "aaaaaaaaa". The issue there is that you need to
#   keep changing the palindrome center. I gave up on that approach and looked
#   at the solution.
# - The solution is straightforward and I probably would have thought of it, if
#   I didn't get stuck trying to fix the three pointer approach.
def longestPalindrome(s: str) -> str:
    """
    Examples:
    >>> longestPalindrome("babad")
    'bab'
    >>> longestPalindrome("cbbd")
    'bb'
    >>> longestPalindrome("ac")
    'a'
    >>> longestPalindrome("abcde")
    'a'
    >>> longestPalindrome("abcdeedcba")
    'abcdeedcba'
    >>> longestPalindrome("abcdeeffdcba")
    'ee'
    >>> longestPalindrome("abaaba")
    'abaaba'
    >>> longestPalindrome("abaabac")
    'abaaba'
    >>> longestPalindrome("aaaaa")
    'aaaaa'
    >>> longestPalindrome("aaaa")
    'aaaa'
    """
    if len(s) == 1:
        return s

    lo, hi = 0, 1
    max_length = 1
    res_string = s[0]

    def expand_center(lo, hi):
        while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
            lo -= 1
            hi += 1
        return lo + 1, hi - 1

    for i in range(1, len(s)):
        lo, hi = expand_center(i - 1, i + 1)
        if hi - lo + 1 > max_length:
            max_length = hi - lo + 1
            res_string = s[lo : hi + 1]

        lo, hi = expand_center(i - 1, i)
        if hi - lo + 1 > max_length:
            max_length = hi - lo + 1
            res_string = s[lo : hi + 1]

    return res_string
