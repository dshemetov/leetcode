# %% 18. 4Sum https://leetcode.com/problems/4sum/


# Lessons learned:
# - The idea is the same as in 3Sum, but with an extra index.
def four_sum(nums: list[int], target: int) -> list[list[int]]:
    """
    Examples:
    >>> four_sum([1,0,-1,0,-2,2], 0)
    [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
    >>> four_sum([2,2,2,2,2], 8)
    [[2, 2, 2, 2]]
    >>> four_sum([-2,-1,-1,1,1,2,2], 0)
    [[-2, -1, 1, 2], [-1, -1, 1, 1]]
    """
    nums.sort()
    res = []
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            lo, hi = j + 1, len(nums) - 1
            while lo < hi:
                s = nums[i] + nums[j] + nums[lo] + nums[hi]
                if s < target:
                    lo += 1
                elif s > target:
                    hi -= 1
                else:
                    res.append([nums[i], nums[j], nums[lo], nums[hi]])

                    while lo < hi and nums[lo] == nums[lo + 1]:
                        lo += 1
                    while lo < hi and nums[hi] == nums[hi - 1]:
                        hi -= 1

                    lo += 1
                    hi -= 1

    return res
