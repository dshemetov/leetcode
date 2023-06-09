# %% 15. 3Sum https://leetcode.com/problems/3sum/
def three_sum(nums: list[int]) -> list[list[int]]:
    """
    Examples:
    >>> three_sum([-1,0,1,2,-1,-4])
    [[-1, -1, 2], [-1, 0, 1]]
    >>> three_sum([0, 1, 1])
    []
    >>> three_sum([0, 0, 0])
    [[0, 0, 0]]
    """
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:
                lo += 1
            elif s > 0:
                hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])

                while lo < hi and nums[lo] == nums[lo + 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi - 1]:
                    hi -= 1

                lo += 1
                hi -= 1

    return res
