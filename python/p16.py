def p1544(s: str) -> str:
    """
    1544. Make The String Great https://leetcode.com/problems/make-the-string-great/

    Examples:
    >>> p1544("leEeetcode")
    'leetcode'
    >>> p1544("abBAcC")
    ''
    >>> p1544("s")
    's'
    """
    stack = []
    for c in s:
        if stack and stack[-1].lower() == c.lower() and stack[-1] != c:
            stack.pop()
        else:
            stack.append(c)
    return "".join(stack)


def p1557(n: int, edges: list[list[int]]) -> list[int]:
    """
    1557. Minimum Number of Vertices to Reach All Nodes https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/

    Lessons learned:
    - At first I thought this required union find, but that is for partitions /
    undirected graphs. After fiddling with a modification of union find for a
    while, I saw that the solution was actually really simple.

    Examples:
    >>> p1557(6, [[0,1],[0,2],[2,5],[3,4],[4,2]])
    [0, 3]
    >>> p1557(5, [[0,1],[2,1],[3,1],[1,4],[2,4]])
    [0, 2, 3]
    """
    nodes_with_parents = set()
    for _, v in edges:
        nodes_with_parents.add(v)

    return [i for i in range(n) if i not in nodes_with_parents]


def p1572(mat: list[list[int]]) -> int:
    """
    1572. Matrix Diagonal Sum https://leetcode.com/problems/matrix-diagonal-sum/

    Examples:
    >>> p1572([[1,2,3],[4,5,6],[7,8,9]])
    25
    >>> p1572([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    8
    """
    n = len(mat)

    if n == 1:
        return mat[0][0]

    total = 0

    for i in range(n):
        total += mat[i][i] + mat[n - 1 - i][i]

    if n % 2 == 1:
        total -= mat[n // 2][n // 2]

    return total


def p1579(n: int, edges: list[list[int]]) -> int:
    """
    1579. Remove Max Number of Edges to Keep Graph Fully Traversable https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/

    Lessons learned:
    - We can build a spanning tree greedily by adding edges when they don't create
    a cycle. We can detect when an edge would create a cycle, by using a
    disjoint set. Counting these edges gives us the number removable edges. This
    problem adds a minor complication by having three types of edges. This
    complication can be dealth with by keeping track of two graphs. Since
    sometimes one edge of type 3 can make two edges of type 1 and 2 obsolete, we
    prioritize adding edges of type 3 first.
    - A spanning tree always has the minimum number of edges to connect all nodes,
    which is V - 1 for a graph with V nodes

    Examples:
    >>> p1579(4, [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]])
    2
    >>> p1579(4, [[3,1,2],[3,2,3],[1,1,4],[2,1,4]])
    0
    >>> p1579(4, [[3,2,3],[1,1,2],[2,3,4]])
    -1
    >>> p1579(2, [[1,1,2],[2,1,2],[3,1,2]])
    2
    """

    def find(x: int, parent: list[int]) -> int:
        while True:
            if parent[x] == x:
                return x
            parent[x] = parent[parent[x]]
            x = parent[x]

    def union(x: int, y: int, parent: list[int]) -> bool:
        """Return True if new connection made."""
        x_root, y_root = find(x, parent), find(y, parent)
        if x_root == y_root:
            return False
        parent[x_root] = y_root
        return True

    alice_graph = list(range(n))
    bob_graph = list(range(n))
    total_edges = 0
    for edge_type, s, t in edges:
        if edge_type == 3:
            ag = union(s - 1, t - 1, alice_graph)
            bg = union(s - 1, t - 1, bob_graph)
            if not (ag or bg):
                total_edges += 1

    for edge_type, s, t in edges:
        if edge_type == 1:
            if not union(s - 1, t - 1, alice_graph):
                total_edges += 1
        elif edge_type == 2:
            if not union(s - 1, t - 1, bob_graph):
                total_edges += 1
        else:
            continue

    def count(parent: list[int]) -> int:
        return len({find(i, parent) for i in range(n)})

    if count(alice_graph) > 1 or count(bob_graph) > 1:
        return -1

    return total_edges
