# %% 495. Teemo Attacking https://leetcode.com/problems/teemo-attacking
def findPoisonedDuration(timeSeries: list[int], duration: int) -> int:
    """
    Examples:
    >>> findPoisonedDuration([1,4], 2)
    4
    >>> findPoisonedDuration([1,2], 2)
    3
    """
    total_duration = 0
    for i in range(1, len(timeSeries)):
        time_delta = timeSeries[i] - timeSeries[i - 1]
        total_duration += min(duration, time_delta)
    return total_duration + duration
