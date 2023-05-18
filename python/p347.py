# 347. Top K Frequent Elements https://leetcode.com/problems/top-k-frequent-elements/
# Lessons learned:
from typing import Counter


def topKFrequent(nums: list[int], k: int) -> list[int]:
    """
    Examples:
    >>> topKFrequent([1,1,1,2,2,3], 2)
    [1, 2]
    >>> topKFrequent([1], 1)
    [1]
    """
    c = Counter(nums)
    return [num for num, _ in c.most_common(k)]
