# %% 242. Valid Anagram https://leetcode.com/problems/valid-anagram/
from collections import Counter


def isAnagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)
