# %% 3. Longest Substring Without Repeating Characters https://leetcode.com/problems/longest-substring-without-repeating-characters/
def length_of_longest_substring(s: str) -> int:
    """
    Examples:
    >>> length_of_longest_substring("a")
    1
    >>> length_of_longest_substring("aa")
    1
    >>> length_of_longest_substring("aaa")
    1
    >>> length_of_longest_substring("aab")
    2
    >>> length_of_longest_substring("abba")
    2
    >>> length_of_longest_substring("abccba")
    3
    >>> length_of_longest_substring("au")
    2
    >>> length_of_longest_substring("cdd")
    2
    >>> length_of_longest_substring("abcabcbb")
    3
    >>> length_of_longest_substring("aabcdef")
    6
    >>> length_of_longest_substring("abcdeffff")
    6
    >>> length_of_longest_substring("dvdf")
    3
    >>> length_of_longest_substring("ohomm")
    3
    """
    if not s:
        return 0

    longest = 1
    lo = 0
    hi = 1
    char_set = set(s[lo])
    while hi < len(s):
        if s[hi] not in char_set:
            char_set.add(s[hi])
            hi += 1
            longest = max(longest, hi - lo)
        else:
            char_set.remove(s[lo])
            lo += 1

    return longest
