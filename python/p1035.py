# %% 1035. Uncrossed Lines https://leetcode.com/problems/uncrossed-lines/


# Lessons learned:
# - The solution is identical to (1143 Longest Common Subsequence).
def maxUncrossedLines(nums1: list[int], nums2: list[int]) -> int:
    """
    Examples:
    >>> maxUncrossedLines([1,4,2], [1,2,4])
    2
    >>> maxUncrossedLines([2,5,1,2,5], [10,5,2,1,5,2])
    3
    >>> maxUncrossedLines([1,3,7,1,7,5], [1,9,2,5,1])
    2
    """
    dp_ = [[0 for _ in range(len(nums2) + 1)] for _ in range(len(nums1) + 1)]

    for i in range(1, len(nums1) + 1):
        for j in range(1, len(nums2) + 1):
            if nums1[i - 1] == nums2[j - 1]:
                dp_[i][j] = 1 + dp_[i - 1][j - 1]
            else:
                dp_[i][j] = max(dp_[i - 1][j], dp_[i][j - 1])

    return dp_[-1][-1]
