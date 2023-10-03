def p2540(nums1: list[int], nums2: list[int]) -> int:
    """
    2540. Minimum Common Value https://leetcode.com/problems/minimum-common-value/

    Examples:
    >>> p2540([1, 2, 3], [2, 4])
    2
    >>> p2540([1, 2, 3], [4, 5])
    -1
    >>> p2540([1, 2, 3, 6], [2, 3, 4, 5])
    2
    >>> p2540([1, 1, 2], [2, 4])
    2
    """
    n, m = len(nums1), len(nums2)
    i, j = 0, 0
    while i < n and j < m:
        if nums1[i] == nums2[j]:
            return nums1[i]
        if nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1

    return -1
