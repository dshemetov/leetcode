# %% 1345. Jump Game IV https://leetcode.com/problems/jump-game-iv/
from collections import defaultdict
from queue import PriorityQueue


# Lessons learned:
# - I did this with a priority queue, but it's not necessary. A BFS would work
#   just as well.
# - You can also do a bidirectional BFS, which can be faster. This means
#   building a frontier of nodes from both the start and the end.
def minJumps(arr: list[int]) -> int:
    """
    Examples:
    >>> minJumps([100,-23,-23,404,100,23,23,23,3,404])
    3
    >>> minJumps([7])
    0
    >>> minJumps([7,6,9,6,9,6,9,7])
    1
    >>> minJumps([7,7,2,1,7,7,7,3,4,1])
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
