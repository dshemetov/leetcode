# %% 1703. Minimum Adjacent Swaps for K Consecutive Ones https://leetcode.com/problems/minimum-adjacent-swaps-for-k-consecutive-ones/
from collections import deque


# Lesson learned:
# - The first solution is not fast enough, but has the right general idea. It
#   uses a sliding window of size k that records the indices of the 1s and finds
#   the the absolute value of the distances to the median index in that window.
# - The second solution makes two changes. The first is to change focus from
#   storing the indices of the 1s to storing the gaps between the indices of the
#   1s. This is because the absolute value distance to the median has a formula
#   in terms of the differences between the indices. For example, with nums =
#   [1, 1, 0, 0, 1, 0, 1, 0, 1], we have gaps = [1, 3, 2, 2], so for k = 5, the
#   distance to the median (the third 1) is 1 + 2 * 3 + 2 * 2 + 2 = 13. In
#   general, we have
#
#       av(j)   = sum_{i=0}^{k-2} min(i + 1, k - (i + 1)) * gaps[j + i]
#
#   Furthermore, d1(j) = av(j) - av(j-1) and d2(j) = d1(j) - d1(j-1) can be
#   expressed as
#
#       d1(j)   = - sum_{i=0}^{k//2 - 1} gaps[j + i - 1] +                          if k is odd
#                   sum_{i=k//2}^{k-1} gaps[j + i - 1],
#               = - sum_{i=0}^{k//2 - 1} gaps[j + i - 1] +                          else
#                   sum_{i=k//2 + 1}^{k-1} gaps[j + i - 1],
#       d2(j)   = gaps[j] - gaps[j + k//2] - gaps[j + k//2 + 1] + gaps[j + k],      if k is odd
#               = gaps[j] - 2 * gaps[j + k//2] + gaps[j + k],                       else
#
#   This means that if we calculate the first two absolute value distances, we
#   can calculate the subsequent ones via the formula
#
#       av(j)   = av(j-1) + d1(j-1) + d2(j-1),          j >= 2
#       d1(1)   = av(1) - av(0).
#
#   This update step requires us to sum only up to 4 elements from the gaps
#   array, as opposed to k.
# - The absolute value distance needs a correction to equal the swap distance
#
#       av      = swap - T(L) - T(R), where T(x) = x * (x + 1) // 2
#
#   where L, R are number of neighbors on the left and right of the median,
#   respectively. For example, if the k-stack is [1, 1, 0, 0, 1, 0, 1], the
#   absolute value distance to the median is 9, while the actual swaps needed is
#   5. The correction factor can be found T(L) + T(R) = T(2) + T(1) = 3 + 1 = 4.
# - I solved this problem on my own, so I feel pretty proud of it.
# - Taking a look at other people's solutions, I see simpler approaches. For
#   example, it turns out that the absolute value distance to the median can be
#   expressed as
#
#       av(j)   = sum(pos[j+k//2:j+k]) - sum(pos[j:j+k//2])
#       pos(j)  = position of the jth 1 in nums
#
#   from which it would be much easier to calculate the iterative approach. The
#   way to arrive at this, is to notice that the median point has half the 1s to
#   its left and half to its right.
# - There is a competitive programming trick to convert a problem from "min sum
#   of moves to make numbers consecutive" to "max sum of moves to a single
#   point" and that is to transform pos(j) to pos(j) - j. This is because
#
#       sum pos(j) - j = sum pos(j) - sum j = sum pos(j) - k * (k + 1) // 2
#
#   which is the correction factor we discused above.
def get_gaps(nums: list[int]) -> list[int]:
    gaps = []
    prev = -1
    for i, x in enumerate(nums):
        if prev == -1 and x == 1:
            prev = i
        elif prev != -1 and x == 1:
            gaps.append(i - prev)
            prev = i
    return gaps


def minMoves(nums: list[int], k: int) -> int:
    """
    Examples:
    >>> minMoves([1,0,0,1,0,1], 2)
    1
    >>> minMoves([1,0,0,0,0,0,1,1], 3)
    5
    >>> minMoves([1,1,0,1], 2)
    0
    >>> minMoves([0,0,0,1,0,1,1,0,1], 3)
    1
    >>> minMoves([0,0,0,1,0,1,1,0,1], 4)
    2
    >>> minMoves([1,0,1,0,1,0,0,0,1], 4)
    6
    """
    if len(nums) == k or k == 1:
        return 0

    def triangular_number(n: int) -> int:
        return n * (n + 1) // 2

    mid1, mid2 = k // 2 - 1, k // 2
    if k % 2 == 0:
        correction_factor = triangular_number(mid1) + triangular_number(mid2)
    else:
        correction_factor = 2 * triangular_number(mid2)

    def calculate_swaps(stack: deque[int]) -> int:
        """Calculate the swaps needed to get to the median(s) of the stack.

        The stack represents the indices of the 1s in the array.
        """
        return sum(abs(stack[mid2] - i) for i in stack)

    stack = deque()
    s = 0
    m = float("inf")
    for i, n in enumerate(nums):
        if n == 1:
            stack.append(i)
            if stack and len(stack) > k:
                s = s - stack.popleft()
            if len(stack) == k:
                m = min(m, calculate_swaps(stack))

    return m - correction_factor


def minMoves2(nums: list[int], k: int) -> int:
    """
    Examples:
    >>> minMoves2([1,0,0,1,0,0], 2)
    2
    >>> minMoves2([1,0,0,1,0,1], 2)
    1
    >>> minMoves2([1,0,0,0,0,0,1,1], 3)
    5
    >>> minMoves2([1,1,0,1], 2)
    0
    >>> minMoves2([0,0,0,1,0,1,1,0,1], 3)
    1
    >>> minMoves2([0,0,0,1,0,1,1,0,1], 4)
    2
    >>> minMoves2([1,0,1,0,1,0,0,0,1], 4)
    6
    """
    if len(nums) == k or k == 1:
        return 0

    def triangular_number(n: int) -> int:
        return n * (n + 1) // 2

    mid1, mid2 = k // 2 - 1, k // 2
    if k % 2 == 0:
        correction_factor = triangular_number(mid1) + triangular_number(mid2)
    else:
        correction_factor = 2 * triangular_number(mid2)

    def calc2(gaps: list, k: int, j: int) -> int:
        """Calculate using the absolute value formula with the differences between indices."""
        return sum(min(i + 1, k - (i + 1)) * gaps[j + i] for i in range(k - 1))

    gaps = get_gaps(nums)

    if len(gaps) == k - 1:
        return calc2(gaps, k, 0) - correction_factor

    m0, m1 = calc2(gaps, k, 0), calc2(gaps, k, 1)
    delta1 = m1 - m0
    m = min(m0, m1)

    for j in range(0, len(gaps) - k):
        if k % 2 == 1:
            delta2 = gaps[j] - gaps[j + k // 2] - gaps[j + k // 2 + 1] + gaps[j + k]
        else:
            delta2 = gaps[j] - 2 * gaps[j + k // 2] + gaps[j + k]
        delta1 = delta1 + delta2
        m1 = m1 + delta1
        m = min(m, m1)

    return m - correction_factor


def get_moves_list(nums: list[int], k: int) -> int:
    """Test three ways to calculate the absolute value distance.

    Examples:
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 2)
    >>> a == b == c
    True
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 3)
    >>> a == b == c
    True
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 4)
    >>> a == b == c
    True
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 5)
    >>> a == b == c
    True
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 6)
    >>> a == b == c
    True
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 8)
    >>> a == b == c
    True
    """

    def calc1(stack: deque, k: int) -> int:
        """Calculate using the absolute value formula with the indices."""
        mid = k // 2
        return sum(abs(stack[mid] - i) for i in stack)

    def method1(nums: list[int], k: int) -> list[int]:
        """Calculate using the absolute value formula with the indices."""
        if len(nums) == k or k == 1:
            return 0

        moves = []
        stack = deque()
        s = 0
        for i, n in enumerate(nums):
            if n == 1:
                stack.append(i)
                if stack and len(stack) > k:
                    s = s - stack.popleft()
                if len(stack) == k:
                    moves.append(calc1(stack, k))

        return moves

    def calc2(gaps: list, k: int, j: int) -> int:
        """Calculate using the absolute value formula with the differences between indices."""
        return sum(min(i + 1, k - (i + 1)) * gaps[j + i] for i in range(k - 1))

    def method2(nums: list[int], k: int) -> list[int]:
        """Calculate using the absolute value formula with the differences between indices."""
        gaps = get_gaps(nums)
        moves = []
        for j in range(len(gaps) - k + 2):
            moves.append(calc2(gaps, k, j))
        return moves

    def method3(nums: list[int], k: int) -> list[int]:
        """Calculate using a magic iterative approach."""
        gaps = get_gaps(nums)

        if len(gaps) == k - 1:
            return [calc2(gaps, k, 0)]

        moves = [calc2(gaps, k, 0), calc2(gaps, k, 1)]
        delta1 = moves[-1] - moves[-2]
        for j in range(0, len(gaps) - k):
            if k % 2 == 1:
                delta2 = gaps[j] - gaps[j + k // 2] - gaps[j + k // 2 + 1] + gaps[j + k]
            else:
                delta2 = gaps[j] - 2 * gaps[j + k // 2] + gaps[j + k]
            delta1 = delta1 + delta2
            moves.append(moves[-1] + delta1)
        return moves

    return method1(nums, k), method2(nums, k), method3(nums, k)
