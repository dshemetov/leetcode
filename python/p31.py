def p3005(nums: list[int]) -> int:
    """
    3005. Count Elements With Maximum Frequency https://leetcode.com/problems/count-elements-with-maximum-frequency

    Examples:
    >>> p3005([1, 2, 2, 3, 1, 4])
    4
    >>> p3005([1, 2, 3, 4, 5])
    5
    """
    from collections import Counter

    c = Counter(nums)
    max_value = max(c.values())
    return sum(v for v in c.values() if v == max_value)
