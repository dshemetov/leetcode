# %% 399. Evaluate Division https://leetcode.com/problems/evaluate-division/
from collections import defaultdict, deque


def calcEquation(
    equations: list[list[str]], values: list[float], queries: list[list[str]]
) -> list[float]:
    """
    Examples:
    >>> calcEquation([["a","b"],["b","c"]], [2.0,3.0], [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]])
    [6.0, 0.5, -1.0, 1.0, -1.0]
    >>> calcEquation([["a","b"],["b","c"],["bc","cd"]], [1.5,2.5,5.0], [["a","c"],["c","b"],["bc","cd"],["cd","bc"]])
    [3.75, 0.4, 5.0, 0.2]
    >>> calcEquation([["a","b"]], [0.5], [["a","b"],["b","a"],["a","c"],["x","y"]])
    [0.5, 2.0, -1.0, -1.0]
    """
    graph: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), v in zip(equations, values):
        graph[a][b] = v
        graph[b][a] = 1 / v

    def dfs(a: str, b: str) -> float:
        if a not in graph or b not in graph:
            return -1.0
        unexplored = deque([(a, 1.0)])
        visited = set()
        while unexplored:
            node, cost = unexplored.pop()
            visited.add(node)
            if node == b:
                return cost
            for child in graph[node]:
                if child not in visited:
                    unexplored.append((child, cost * graph[node][child]))
        return -1.0

    return [dfs(a, b) for a, b in queries]
