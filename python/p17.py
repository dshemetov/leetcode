import math


def p1680(n: int) -> int:
    """
    1680. Concatenation of Consecutive Binary Numbers https://leetcode.com/problems/concatenation-of-consecutive-binary-numbers/

    Examples:
    >>> p1680(1)
    1
    >>> p1680(3)
    27
    >>> p1680(12)
    505379714
    """
    M = 10**9 + 7
    total = 1
    for i in range(2, n + 1):
        total = ((total << math.floor(math.log2(i)) + 1) + i) % M

    return total


def p1697(n: int, edgeList: list[list[int]], queries: list[list[int]]) -> list[bool]:
    """
    1697. Checking Existence of Edge Length Limited Paths https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/

    Lessons learned:
    - This problem is a connected component problem, though the weighted edges may
    throw you off. Since we're not looking for total path distance, for each
    query in order of increasing threshold, we can build a graph and calculate
    the connected components given by the query threshold. This lets us build on
    the work done for previous queries.

    Examples:
    >>> p1697(3, [[0,1,2],[1,2,4],[2,0,8],[1,0,16]], [[0,1,2],[0,2,5]])
    [False, True]
    >>> p1697(5, [[0,1,10],[1,2,5],[2,3,9],[3,4,13]], [[0,4,14],[1,4,13]])
    [True, False]
    >>> p1697(3, [[0,1,2],[1,2,4],[2,0,8],[1,0,16]], [[0,2,1], [0,2,7]])
    [False, True]
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while True:
            if x == parent[x]:
                return x
            parent[x] = parent[parent[x]]
            x = parent[x]

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    edgeList.sort(key=lambda x: x[2])
    queries = sorted((q[2], q[0], q[1], i) for i, q in enumerate(queries))

    result = [False] * len(queries)
    i = 0
    for d, q0, q1, j in queries:
        while i < len(edgeList) and edgeList[i][2] < d:
            union(edgeList[i][0], edgeList[i][1])
            i += 1
        result[j] = find(q0) == find(q1)

    return result
