# %% 1491. Average Salary Excluding the Minimum and Maximum Salary https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/


# Lessons learned:
# - Slightly surprised the single pass Python-loop approach is slightly faster
#   than the three pass approach using built-ins.
def average(salary: list[int]) -> float:
    """
    Examples:
    >>> average([4000,3000,1000,2000])
    2500.0
    >>> average([1000,2000,3000])
    2000.0
    """
    return (sum(salary) - min(salary) - max(salary)) / (len(salary) - 2)


def average2(salary: list[int]) -> float:
    """
    Examples:
    >>> average2([4000,3000,1000,2000])
    2500.0
    >>> average2([1000,2000,3000])
    2000.0
    """
    lo, hi = float("inf"), float("-inf")
    sums = 0
    count = 0
    for s in salary:
        if s < lo:
            lo = s
        if s > hi:
            hi = s
        sums += s
        count += 1

    return (sums - lo - hi) / (count - 2)
