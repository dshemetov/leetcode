def p1456(s: str, k: int) -> int:
    """
    1456. Maximum Number of Vowels in a Substring of Given Length https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/

    Lessons learned:
    - Sliding window and no need for a queue here, because sum statistics are easy
    to update.

    Examples:
    >>> p1456("abciiidef", 3)
    3
    >>> p1456("aeiou", 2)
    2
    >>> p1456("leetcode", 3)
    2
    >>> p1456("rhythms", 4)
    0
    >>> p1456("tryhard", 4)
    1
    """
    vowels = set("aeiou")
    num_vowels = sum(c in vowels for c in s[:k])
    max_vowels = num_vowels
    for i in range(k, len(s)):
        if s[i - k] in vowels:
            num_vowels -= 1
        if s[i] in vowels:
            num_vowels += 1
        max_vowels = max(max_vowels, num_vowels)
    return max_vowels


def p1491(salary: list[int]) -> float:
    """
    1491. Average Salary Excluding the Minimum and Maximum Salary https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/

    Lessons learned:
    - Slightly surprised the single pass Python-loop approach is slightly faster
    than the three pass approach using built-ins.

    Examples:
    >>> p1491([4000,3000,1000,2000])
    2500.0
    >>> p1491([1000,2000,3000])
    2000.0
    """
    return (sum(salary) - min(salary) - max(salary)) / (len(salary) - 2)


def p1491_2(salary: list[int]) -> float:
    """
    Examples:
    >>> p1491_2([4000,3000,1000,2000])
    2500.0
    >>> p1491_2([1000,2000,3000])
    2000.0
    """
    lo, hi = float("inf"), float("-inf")
    sums = 0
    count = 0
    for s in salary:
        if s < lo:
            lo = s
        if s > hi:
            hi = s
        sums += s
        count += 1

    return (sums - lo - hi) / (count - 2)


def p1498(nums: list[int], target: int) -> int:
    """
    1498. Number of Subsequences That Satisfy the Given Sum Condition https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/

    Lessons learned:
    - I had the rough idea, but I was tired, so I looked at a hint.
    - 1 << n is much faster in Python than 2**n.

    Examples:
    >>> p1498([3,5,6,7], 9)
    4
    >>> p1498([3,3,6,8], 10)
    6
    >>> p1498([2,3,3,4,6,7], 12)
    61
    >>> p1498([14,4,6,6,20,8,5,6,8,12,6,10,14,9,17,16,9,7,14,11,14,15,13,11,10,18,13,17,17,14,17,7,9,5,10,13,8,5,18,20,7,5,5,15,19,14], 22)
    272187084
    """
    nums.sort()
    lo, hi = 0, len(nums) - 1
    count = 0
    while lo <= hi:
        if nums[lo] + nums[hi] <= target:
            count += 1 << (hi - lo)
            lo += 1
        else:
            hi -= 1
    return count % (10**9 + 7)
