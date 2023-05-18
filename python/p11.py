# %% 11. Container With Most Water https://leetcode.com/problems/container-with-most-water/


# Lessons learned:
# - The trick to the O(n) solution relies on the following insight: if we
#   shorten the container but change the height of the larger side, the area
#   will not increase. Therefore, we can start with the widest possible
#   container and do at most one comparison per index.
# - This feels like a trick problem and I didn't feel like I learned much from
#   it.
def max_area(height: list[int]) -> int:
    """
    Examples:
    >>> max_area([1,8,6,2,5,4,8,3,7])
    49
    >>> max_area([1,1])
    1
    """
    lo, hi = 0, len(height) - 1
    m = float("-inf")
    while lo < hi:
        m = max(m, min(height[lo], height[hi]) * (hi - lo))
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1
    return m
