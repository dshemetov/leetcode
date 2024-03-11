from collections import deque
from collections import defaultdict


class p901:
    """
    901. Online Stock Span https://leetcode.com/problems/online-stock-span/

    Lessons learned:
    - This uses a monotonically decreasing stack (MDS) to keep track of the
    previous stock prices and their spans.

    Examples:
    >>> obj = p901()
    >>> obj.next(100)
    1
    >>> obj.next(80)
    1
    >>> obj.next(60)
    1
    >>> obj.next(70)
    2
    >>> obj.next(60)
    1
    >>> obj.next(75)
    4
    >>> obj.next(85)
    6
    """

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        self.stack.append([price, span])
        return span


def p934(grid: list[list[int]]) -> int:
    """
    934. Shortest Bridge https://leetcode.com/problems/shortest-bridge/

    Lessons learned:
    - This problem has a couple sub-problems that allow for variants.
    - First, there is the problem of correctly coloring the connected components.
    This can be done with a simple DFS and an extra coloring dictionary, but
    here we modify the input grid to save space.
    - Second, there is the path-finding problem. This can be done with BFS.

    Examples:
    >>> p934([[0,1],[1,0]])
    1
    >>> p934([[0,1,0],[0,0,0],[0,0,1]])
    2
    >>> p934([[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]])
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


def p947(stones: list[list[int]]) -> int:
    """
    947. Most Stones Removed With Same Row or Column https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/

    Lessons learned:
    - The key idea is that we can remove all stones in each connected component
    except one. We can use dfs to find the connected components. Fun fact: the
    dfs can avoid recursion by using a stack.

    Examples:
    >>> p947([[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]])
    5
    >>> p947([[0,0],[0,2],[1,1],[2,0],[2,2]])
    3
    >>> p947([[0,0]])
    0
    >>> p947([[0,0],[0,1],[1,1]])
    2
    >>> p947([[0,1],[1,0]])
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


def p977(nums: list[int]) -> list[int]:
    """
    977. Squares of a Sorted Array https://leetcode.com/problems/squares-of-a-sorted-array/

    Examples:
    >>> p977([-4,-1,0,3,10])
    [0, 1, 9, 16, 100]
    >>> p977([-7,-3,2,3,11])
    [4, 9, 9, 49, 121]
    >>> p977([-5,-3,-2,-1])
    [1, 4, 9, 25]
    """
    l, r = 0, len(nums) - 1
    res = [0] * len(nums)
    i = len(nums) - 1
    while l <= r:
        left, right = nums[l] ** 2, nums[r] ** 2
        if left > right:
            res[i] = left
            l += 1
        else:
            res[i] = right
            r -= 1
        i -= 1

    return res


def p990(equations: list[str]) -> bool:
    """
    990. Satisfiability of Equality Equations https://leetcode.com/problems/satisfiability-of-equality-equations/

    Lessons learned:
    - This was clearly a graph problem underneath, where you need to find the
    connected components given by the equality statements
    - Efficiently calculating the connected components was hard for me though, so
    learning about the disjoint set data structure was key (also referred to as
    union find):
    https://cp-algorithms.com/data_structures/disjoint_set_union.html

    Examples:
    >>> assert p990(["a==b", "b!=a"]) is False
    >>> assert p990(["x==y", "z==w", "y==z", "a==b", "d==e", "f==g", "e==f", "w==x", "c==d", "b==d", "g!=x"]) is True
    >>> assert p990(["x==y", "z==w", "y==z", "a==b", "d==e", "f==g", "e==f", "w==x", "c==d", "b==d", "g!=x", "a==z"]) is False
    >>> assert p990(["x==a", "w==b", "z==c", "a==b", "b==c", "c!=x"]) is False
    >>> assert p990(["a==b", "c==e", "b==c", "a!=e"]) is False
    >>> assert p990(["a==b", "e==c", "c==b", "a!=e"]) is False
    >>> assert p990(["a==b", "e==c", "c==b", "a!=e"]) is False
    >>> assert p990(["a==b", "e==c", "b==c", "a!=e"]) is False
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while True:
            if parent[x] == x:
                return x
            parent[x] = parent[parent[x]]
            x = parent[x]

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for x, eq, _, y in equations:
        if eq == "=":
            parent.setdefault(x, x)
            parent.setdefault(y, y)
            union(x, y)

    for x, eq, _, y in equations:
        if eq == "!":
            if x == y:
                return False
            if find(x) == find(y):
                return False
    return True
