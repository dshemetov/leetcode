# %% 947. Most Stones Removed With Same Row or Column https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/
from collections import defaultdict


# Lessons learned:
# - The key idea is that we can remove all stones in each connected component
#   except one. We can use dfs to find the connected components. Fun fact: the
#   dfs can avoid recursion by using a stack.
def removeStones(stones: list[list[int]]) -> int:
    """
    Examples:
    >>> removeStones([[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]])
    5
    >>> removeStones([[0,0],[0,2],[1,1],[2,0],[2,2]])
    3
    >>> removeStones([[0,0]])
    0
    >>> removeStones([[0,0],[0,1],[1,1]])
    2
    >>> removeStones([[0,1],[1,0]])
    0
    """
    rows = defaultdict(list)
    cols = defaultdict(list)

    for i, (r, c) in enumerate(stones):
        rows[r].append(i)
        cols[c].append(i)

    seen = set()

    def dfs(i: int) -> None:
        """dfs without recursion"""
        stack = [i]
        while stack:
            j = stack.pop()
            seen.add(j)
            for k in rows[stones[j][0]] + cols[stones[j][1]]:
                if k not in seen:
                    stack.append(k)

    n_components = 0
    for i in range(len(stones)):
        if i not in seen:
            dfs(i)
            n_components += 1

    return len(stones) - n_components
