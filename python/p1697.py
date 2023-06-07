# %% 1697. Checking Existence of Edge Length Limited Paths https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/


# Lessons learned:
# - This problem is a connected component problem, though the weighted edges may
#   throw you off. Since we're not looking for total path distance, for each
#   query in order of increasing threshold, we can build a graph and calculate
#   the connected components given by the query threshold. This lets us build on
#   the work done for previous queries.
def distanceLimitedPathsExist(
    n: int, edgeList: list[list[int]], queries: list[list[int]]
) -> list[bool]:
    """
    Examples:
    >>> distanceLimitedPathsExist(3, [[0,1,2],[1,2,4],[2,0,8],[1,0,16]], [[0,1,2],[0,2,5]])
    [False, True]
    >>> distanceLimitedPathsExist(5, [[0,1,10],[1,2,5],[2,3,9],[3,4,13]], [[0,4,14],[1,4,13]])
    [True, False]
    >>> distanceLimitedPathsExist(3, [[0,1,2],[1,2,4],[2,0,8],[1,0,16]], [[0,2,1], [0,2,7]])
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
