# %% 49. Group Anagrams https://leetcode.com/problems/group-anagrams/
from collections import Counter, defaultdict


def groupAnagrams(strs: list[str]) -> list[list[str]]:
    def group_key(s: str) -> Counter:
        return tuple(sorted(s))

    groups = defaultdict(list)
    for s in strs:
        groups[group_key(s)].append(s)

    return list(groups.values())
