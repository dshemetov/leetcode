# %% 4. Median of Two Sorted Arrays https://leetcode.com/problems/median-of-two-sorted-arrays/


# Lessons learned:
# - I spent weeks thinking about this problem before giving up and looking for a
#   solution.
# - There are a few key insights to this problem. First, the median has the
#   property of being the partition point where half the elements are less and
#   half are greater. Second, a partition point in one array implies a partition
#   point in the other array, which means we can find the partition point via
#   binary search on one array.
# - We use the following notation in the code:
#
#       A refers to the shorter array,
#       B refers to the longer array,
#       midA refers to a partition point in A,
#       midB refers to a partition point in B,
#       Aleft = A[midA - 1], refers to the largest element in the left partition of A
#       Aright = A[midA], refers to the smallest element in the right partition of A
#       Bleft = B[midB - 1], refers to the largest element in the left partition of B
#       Bright = B[midB], refers to the smallest element in the right partition of B
#
# - To expand more on the second insight, consider the following example:
#
#      A = [1, 3, 5, 7, 9], B = [2, 4, 6, 8, 10, 12, 14, 16]
#
#   Suppose we choose midA = 4. Since the total number of elements is 13, half
#   of which is 6.5, then, breaking the tie arbitrarily, 7 elements must be in
#   the left partition and 6 elements must be in the right partition. Since 4
#   elements are already in the left partition, we need to add 3 more elements
#   to the left partition, which we can do choosing midB = 3. This corresponds
#   to the total left partition [1, 2, 3, 4, 5, 6, 7] and the total right
#   partition [8, 9, 10, 12, 14, 16].
# - In general, we have
#
#       midA + midB = (len(A) + len(B) + 1) // 2,
#
#   which implies
#
#       midB = (len(A) + len(B) + 1) // 2 - midA.
#
# - Note that having the +1 inside the divfloor covers the cases correctly for
#   odd and even total number of elements. For example, if the total number of
#   elements is 13 and i = 4, then j = (13 + 1) // 2 - 4 = 3, which is correct.
#   If the total number of elements is 12 and i = 4, then j = (12 + 1) // 2 - 4
#   = 2, which is also correct. If the +1 was not inside the divfloor, then the
#   second case would be incorrect.
# - So our problem is solved if we can find a partition (midA, midB) with:
#
#       len(A[:midA]) + len(B[:midB]) == len(A[midA:]) + len(B[midB:]),
#       Aleft <= Bright,
#       Bleft <= Aright.
#
# - The median is then
#
#       median = max(Aleft, Bleft)                               if len(A) + len(B) odd
#              = (max(Aleft, Bleft) + min(Aright, Bright)) / 2.  else
#
# - Swapping two variables in Python swaps pointers under the hood:
#   https://stackoverflow.com/a/62038590/4784655.
def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    """
    Examples:
    >>> find_median_sorted_arrays([1, 3], [2])
    2.0
    >>> find_median_sorted_arrays([1, 2], [3, 4])
    2.5
    >>> find_median_sorted_arrays([1, 3], [2, 4])
    2.5
    >>> a1 = [5, 13, 15]
    >>> b1 = [0, 10, 10, 15, 20, 20, 25]
    >>> find_median_sorted_arrays(a1, b1) == get_median_sorted(sorted(a1 + b1))
    True
    >>> a2 = [9, 36, 44, 45, 51, 67, 68, 69]
    >>> b2 = [7, 20, 26, 27, 30, 43, 54, 73, 76, 88, 91, 94]
    >>> find_median_sorted_arrays(a2, b2) == get_median_sorted(sorted(a2 + b2))
    True
    >>> a2 = [2, 2, 2, 2, 2, 2, 5]
    >>> b2 = [0, 0, 0, 0, 0, 0, 0, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    >>> find_median_sorted_arrays(a2, b2) == get_median_sorted(sorted(a2 + b2))
    True
    >>> a2 = [2, 2, 2, 4, 5, 7, 8, 9]
    >>> b2 = [1, 1, 1, 1, 1, 3, 6, 10, 11, 11, 11, 11]
    >>> find_median_sorted_arrays(a2, b2) == get_median_sorted(sorted(a2 + b2))
    True
    """
    if not nums1:
        return get_median_sorted(nums2)
    if not nums2:
        return get_median_sorted(nums1)

    A, B = nums1, nums2

    if len(A) > len(B):
        A, B = B, A

    total = len(A) + len(B)
    lo, hi = 0, len(A)
    while True:
        midA = (lo + hi) // 2
        midB = (total + 1) // 2 - midA

        Aleft = A[midA - 1] if midA > 0 else float("-inf")
        Aright = A[midA] if midA < len(A) else float("inf")
        Bleft = B[midB - 1] if midB > 0 else float("-inf")
        Bright = B[midB] if midB < len(B) else float("inf")

        if Aleft <= Bright and Bleft <= Aright:
            if total % 2 == 0:
                return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
            return float(max(Aleft, Bleft))

        if Aleft > Bright:
            hi = midA - 1
        else:
            lo = midA + 1


def get_median_sorted(nums: list[int]) -> float:
    if len(nums) == 1:
        return nums[0]

    mid = len(nums) // 2

    if len(nums) % 2 == 0:
        return (nums[mid] + nums[mid - 1]) / 2

    return nums[mid]
