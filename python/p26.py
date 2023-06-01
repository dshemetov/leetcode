# %% 26. Remove Duplicates from Sorted Array https://leetcode.com/problems/remove-duplicates-from-sorted-array/
def remove_duplicates(nums: list[int]) -> int:
    """
    Examples:
    >>> remove_duplicates([1, 1, 2])
    2
    >>> remove_duplicates([0,0,1,1,1,2,2,3,3,4])
    5
    """
    k = 0
    for i in range(1, len(nums)):
        if nums[k] != nums[i]:
            k += 1
            nums[k] = nums[i]
    return k + 1
