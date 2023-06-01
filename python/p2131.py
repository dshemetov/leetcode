# %% 2131. Longest Palindrome by Concatenating Two Letter Words https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/
from collections import Counter


def longestPalindrome2(words: list[str]) -> int:
    """
    Examples:
    >>> longestPalindrome2(["ab","ba","aa","bb","cc"])
    6
    >>> longestPalindrome2(["ab","ba","cc","ab","ba","cc"])
    12
    >>> longestPalindrome2(["aa","ba"])
    2
    >>> longestPalindrome2(["ba", "ce"])
    0
    >>> longestPalindrome2(["lc","cl","gg"])
    6
    >>> longestPalindrome2(["ab","ty","yt","lc","cl","ab"])
    8
    >>> longestPalindrome2(["cc","ll","xx"])
    2
    """
    d: Counter[str] = Counter()

    for word in words:
        d[word] += 1

    total = 0
    extra_double = False
    for key in d:
        if key[0] == key[1]:
            total += d[key] // 2 * 4
            if d[key] % 2 == 1:
                extra_double = True
        elif key == "".join(sorted(key)):
            total += min(d[key], d[key[::-1]]) * 4

    if extra_double:
        total += 2

    return total
