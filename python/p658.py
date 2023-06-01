# %% 658. Find k Closest Elements https://leetcode.com/problems/find-k-closest-elements/


# Lessons learned:
# - My solution uses a straightforward binary search to find the closest element
#   to x and iterated from there.
# - I include a clever solution from the discussion that uses binary search to
#   find the leftmost index of the k closest elements.
# - I had some vague intuition that it could be framed as a minimization
#   problem, but I couldn't find the loss function.
def findClosestElements(arr: list[int], k: int, x: int) -> list[int]:
    """
    Examples:
    >>> findClosestElements([1, 2, 3, 4, 5], 4, 3)
    [1, 2, 3, 4]
    >>> findClosestElements([1, 2, 3, 4, 5], 4, -1)
    [1, 2, 3, 4]
    >>> findClosestElements([1, 2, 3, 4, 5], 4, 4)
    [2, 3, 4, 5]
    >>> findClosestElements([1, 2, 3, 4, 5], 2, 4)
    [3, 4]
    """

    def find_insertion_index(arr: list[int], x: int) -> int:
        lo, hi = 0, len(arr) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == x:
                return mid
            if arr[mid] < x:
                lo = mid + 1
            hi = mid - 1
        return lo

    ix = find_insertion_index(arr, x)
    lst = []
    if ix == 0:
        lst = arr[:k]
    elif ix == len(arr):
        lst = arr[-k:]
    else:
        lo, hi = ix - 1, ix

        while len(lst) < k:
            if lo < 0:
                lst.append(arr[hi])
                hi += 1
            elif hi >= len(arr):
                lst.append(arr[lo])
                lo -= 1
            elif abs(x - arr[lo]) <= abs(x - arr[hi]):
                lst.append(arr[lo])
                lo -= 1
            elif abs(x - arr[lo]) > abs(x - arr[hi]):
                lst.append(arr[hi])
                hi += 1

    return sorted(lst)


def findClosestElements2(arr: list[int], k: int, x: int) -> list[int]:
    """
    Examples:
    >>> findClosestElements2([1, 2, 3, 4, 5], 4, 3)
    [1, 2, 3, 4]
    >>> findClosestElements2([1, 2, 3, 4, 5], 4, -1)
    [1, 2, 3, 4]
    >>> findClosestElements2([1, 2, 3, 4, 5], 4, 4)
    [2, 3, 4, 5]
    >>> findClosestElements2([1, 2, 3, 4, 5], 2, 4)
    [3, 4]
    >>> findClosestElements2([1, 2, 3, 3, 4, 5, 90, 100], 3, 4)
    [3, 3, 4]
    """
    lo, hi = 0, len(arr) - k
    while lo < hi:
        mid = (lo + hi) // 2
        # Equivalently x > (arr[mid] + arr[mid + k]) / 2
        if x - arr[mid] > arr[mid + k] - x:
            lo = mid + 1
        else:
            hi = mid
    return arr[lo : lo + k]
