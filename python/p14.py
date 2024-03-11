import math
from collections import defaultdict
from queue import PriorityQueue


def p1306(arr: list[int], start: int) -> bool:
    """
    1306. Jump Game III https://leetcode.com/problems/jump-game-iii/

    Examples:
    >>> p1306([4,2,3,0,3,1,2], 5)
    True
    >>> p1306([4,2,3,0,3,1,2], 0)
    True
    >>> p1306([3,0,2,1,2], 2)
    False
    """
    seen = set()
    stack = {start}
    while stack:
        ix = stack.pop()

        if arr[ix] == 0:
            return True

        seen.add(ix)

        for ix_ in [ix + arr[ix], ix - arr[ix]]:
            if 0 <= ix_ < len(arr) and ix_ not in seen:
                stack.add(ix_)

    return False


def p1323(num: int) -> int:
    """
    1323. Maximum 69 Number https://leetcode.com/problems/maximum-69-number/

    Lessons learned:
    - Converting to a string and using replace is surprisingly fast.
    - Just need to accept that Python string built-ins are in C-land.

    Examples:
    >>> p1323(9669)
    9969
    >>> p1323(9996)
    9999
    >>> p1323(9999)
    9999
    """
    for i in range(math.floor(math.log10(num)) + 1, -1, -1):
        if num // 10**i % 10 == 6:
            return num + 3 * 10**i
    return num


def p1323_2(num: int) -> int:
    """
    Examples:
    >>> p1323_2(9669)
    9969
    >>> p1323_2(9996)
    9999
    >>> p1323_2(9999)
    9999
    """
    return int(str(num).replace("6", "9", 1))


def p1345(arr: list[int]) -> int:
    """
    1345. Jump Game IV https://leetcode.com/problems/jump-game-iv/

    Lessons learned:
    - I did this with a priority queue, but it's not necessary. A BFS would work
    just as well.
    - You can also do a bidirectional BFS, which can be faster. This means
    building a frontier of nodes from both the start and the end.

    Examples:
    >>> p1345([100,-23,-23,404,100,23,23,23,3,404])
    3
    >>> p1345([7])
    0
    >>> p1345([7,6,9,6,9,6,9,7])
    1
    >>> p1345([7,7,2,1,7,7,7,3,4,1])
    3
    """
    if len(arr) == 1:
        return 0

    value_ix = defaultdict(list)
    for ix, val in enumerate(arr):
        value_ix[val].append(ix)

    seen = set()
    queue = PriorityQueue()
    queue.put((0, 0))
    while queue:
        jumps, ix = queue.get()

        if ix == len(arr) - 1:
            return jumps

        seen.add(ix)

        for ix_ in [ix + 1, ix - 1] + value_ix[arr[ix]]:
            if 0 <= ix_ < len(arr) and ix_ not in seen:
                queue.put((jumps + 1, ix_))

        del value_ix[arr[ix]]

    return -1
