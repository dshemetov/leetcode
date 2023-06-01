# %% 16. 3Sum Closest https://leetcode.com/problems/3sum-closest/
def three_sum_closest(nums: list[int], target: int) -> int:
    """
    Examples:
    >>> three_sum_closest([-1,2,1,-4], 1)
    2
    >>> three_sum_closest([0,0,0], 1)
    0
    """
    nums.sort()
    res = float("inf")
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi] - target
            res = s if abs(s) < abs(res) else res
            if s < 0:
                lo += 1
            elif s > 0:
                hi -= 1
            else:
                return target

    return res + target
