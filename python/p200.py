# %% 200. Number of Islands https://leetcode.com/problems/number-of-islands/
def numIslands(grid: list[list[str]]) -> int:
    n, m = len(grid), len(grid[0])
    visited = set()

    def dfs(i: int, j: int):
        unexplored = {(i, j)}
        while unexplored:
            i_, j_ = unexplored.pop()
            visited.add((i_, j_))

            for i__, j__ in [(i_ + 1, j_), (i_ - 1, j_), (i_, j_ + 1), (i_, j_ - 1)]:
                if (
                    0 <= i__ < n
                    and 0 <= j__ < m
                    and (i__, j__) not in visited
                    and grid[i__][j__] == "1"
                ):
                    unexplored.add((i__, j__))

    islands = 0
    for i in range(n):
        for j in range(m):
            if (i, j) not in visited and grid[i][j] == "1":
                dfs(i, j)
                islands += 1

    return islands
