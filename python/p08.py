def p766(matrix: list[list[int]]) -> bool:
    """
    766. Toeplitz Matrix https://leetcode.com/problems/toeplitz-matrix/

    Examples:
    >>> p766([[1, 2, 3, 4], [5, 1, 2, 3], [9, 5, 1, 2]])
    True
    >>> p766([[1, 2], [2, 2]])
    False
    >>> p766([[11,74,0,93],[40,11,74,7]])
    False
    """
    return all(
        r == 0 or c == 0 or matrix[r - 1][c - 1] == val
        for r, row in enumerate(matrix)
        for c, val in enumerate(row)
    )


def p785(graph: list[list[int]]) -> bool:
    """
    785. Is Graph Bipartite? https://leetcode.com/problems/is-graph-bipartite/

    Lessons learned:
    - A graph is bipartite iff it does not contain any odd cycles. So at first I
    set out to calculate the distances between all nodes and to throw a False if
    I found a loop back to the starting point of odd length. But then I noticed
    that the method I was using was not going to be linear time. I looked up the
    standard method for finding shortest paths between all pairs of nodes in a
    directed, weighted graph (the Floyd-Warshall algorithm), but that was a bit
    overkill too (having a time complexity O(|V|^3)).
    - This problem took over an hour to do. The odd cycles property threw me off,
    making me think that I needed to keep track of node path lengths. Once I let
    go of that idea, I realized that a greedy coloring approach would do the
    trick.

    Examples:
    >>> p785([[1,2,3], [0,2], [0,1,3], [0,2]])
    False
    >>> p785([[1, 3], [0, 2], [1, 3], [0, 2]])
    True
    """
    if not graph:
        return True

    coloring: dict[int, int] = {}

    def dfs(node: int, color: int) -> bool:
        if node in coloring:
            if coloring[node] != color:
                return False
            return True

        coloring[node] = color
        return all(dfs(new_node, color ^ 1) for new_node in graph[node])

    for node in range(len(graph)):
        if node not in coloring and not dfs(node, 0):
            return False

    return True


def p791(order: str, s: str) -> str:
    """
    791. Custom Sort String https://leetcode.com/problems/custom-sort-string/

    Examples:
    >>> p791("cba", "abcd")
    'cbad'
    >>> p791("cba", "abc")
    'cba'
    >>> p791("bcafg", "abcd")
    'bcad'
    """

    def key_fn(t: str) -> int:
        try:
            return order.index(t)
        except ValueError:
            return 30

    return "".join(sorted(s, key=key_fn))
