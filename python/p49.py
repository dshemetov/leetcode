# %% 49. Group Anagrams https://leetcode.com/problems/group-anagrams/
from collections import defaultdict


def groupAnagrams(strs: list[str]) -> list[list[str]]:
    """
    Examples:
    >>> groupAnagrams(["eat","tea","tan","ate","nat","bat"])
    [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
    >>> groupAnagrams([""])
    [['']]
    >>> groupAnagrams(["a"])
    [['a']]
    """

    def group_key(s: str) -> tuple[str, ...]:
        return tuple(sorted(s))

    groups = defaultdict(list)
    for s in strs:
        groups[group_key(s)].append(s)

    return list(groups.values())
