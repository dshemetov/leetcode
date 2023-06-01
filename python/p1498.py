# %% 1498. Number of Subsequences That Satisfy the Given Sum Condition https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/


# Lessons learned:
# - I had the rough idea, but I was tired, so I looked at a hint.
# - 1 << n is much faster in Python than 2**n.
def numSubseq(nums: list[int], target: int) -> int:
    """
    Examples:
    >>> numSubseq([3,5,6,7], 9)
    4
    >>> numSubseq([3,3,6,8], 10)
    6
    >>> numSubseq([2,3,3,4,6,7], 12)
    61
    >>> numSubseq([14,4,6,6,20,8,5,6,8,12,6,10,14,9,17,16,9,7,14,11,14,15,13,11,10,18,13,17,17,14,17,7,9,5,10,13,8,5,18,20,7,5,5,15,19,14], 22)
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
