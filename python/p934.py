# %% 934. Shortest Bridge https://leetcode.com/problems/shortest-bridge/
from collections import deque


# Lessons learned:
# - This problem has a couple sub-problems that allow for variants.
# - First, there is the problem of correctly coloring the connected components.
#   This can be done with a simple DFS and an extra coloring dictionary, but
#   here we modify the input grid to save space.
# - Second, there is the path-finding problem. This can be done with BFS.
def shortestBridge(grid: list[list[int]]) -> int:
    """
    Examples:
    >>> shortestBridge([[0,1],[1,0]])
    1
    >>> shortestBridge([[0,1,0],[0,0,0],[0,0,1]])
    2
    >>> shortestBridge([[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]])
    1
    """
    n = len(grid)
    island1 = deque()

    def dfs(i: int, j: int, color: int) -> None:
        unexplored = deque([(i, j)])
        while unexplored:
            i_, j_ = unexplored.pop()
            grid[i_][j_] = color
            if color == 2:
                island1.append((i_, j_))

            for x, y in [(i_ + 1, j_), (i_ - 1, j_), (i_, j_ + 1), (i_, j_ - 1)]:
                if 0 <= x < n and 0 <= y < n and grid[x][y] == 1:
                    unexplored.append((x, y))

    color = 2
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j, color)
                color += 1
                break
        if color == 4:
            break

    unexplored = island1
    next_unexplored = deque()
    distance = 0
    while True:
        while unexplored:
            i, j = unexplored.pop()

            for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                if 0 <= x < n and 0 <= y < n and grid[x][y] != 2:
                    if grid[x][y] == 3:
                        return distance
                    grid[x][y] = 2
                    next_unexplored.append((x, y))

        unexplored = next_unexplored
        next_unexplored = deque()
        distance += 1
