# %% 345. Reverse Vowels of a String https://leetcode.com/problems/reverse-vowels-of-a-string/
def reverseVowels(s: str) -> str:
    """
    Examples:
    >>> reverseVowels("hello")
    'holle'
    >>> reverseVowels("leetcode")
    'leotcede'
    """
    if len(s) == 1:
        return s

    hi = len(s) - 1
    s_ = []
    for c in s:
        if c in "aeiouAEIOU":
            while s[hi] not in "aeiouAEIOU":
                hi -= 1
            s_.append(s[hi])
            hi -= 1
        else:
            s_.append(c)

    return "".join(s_)
