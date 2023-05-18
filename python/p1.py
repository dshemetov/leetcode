# %% 1. Two Sum https://leetcode.com/problems/two-sum/
from collections import defaultdict


def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Examples:
    >>> two_sum([3, 3], 6)
    [0, 1]
    >>> two_sum([3, 2, 4], 7)
    [0, 2]
    """
    ix_map = defaultdict(list)
    for i, x in enumerate(nums):
        ix_map[x].append(i)

    for x in ix_map:
        if ix_map.get(target - x):
            if x == target - x and len(ix_map.get(x)) == 2:
                return ix_map.get(target - x)
            if x != target - x:
                return [ix_map.get(x)[0], ix_map.get(target - x)[0]]
    return 0
