# %% 1557. Minimum Number of Vertices to Reach All Nodes https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/


# Lessons learned:
# - At first I thought this required union find, but that is for partitions /
#   undirected graphs. After fiddling with a modification of union find for a
#   while, I saw that the solution was actually really simple.
def findSmallestSetOfVertices(n: int, edges: list[list[int]]) -> list[int]:
    """
    Examples:
    >>> findSmallestSetOfVertices(6, [[0,1],[0,2],[2,5],[3,4],[4,2]])
    [0, 3]
    >>> findSmallestSetOfVertices(5, [[0,1],[2,1],[3,1],[1,4],[2,4]])
    [0, 2, 3]
    """
    nodes_with_parents = set()
    for _, v in edges:
        nodes_with_parents.add(v)

    return [i for i in range(n) if i not in nodes_with_parents]
