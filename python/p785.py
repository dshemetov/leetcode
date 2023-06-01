# %% 785. Is Graph Bipartite? https://leetcode.com/problems/is-graph-bipartite/


# Lessons learned:
# - A graph is bipartite iff it does not contain any odd cycles. So at first I
#   set out to calculate the distances between all nodes and to throw a False if
#   I found a loop back to the starting point of odd length. But then I noticed
#   that the method I was using was not going to be linear time. I looked up the
#   standard method for finding shortest paths between all pairs of nodes in a
#   directed, weighted graph (the Floyd-Warshall algorithm), but that was a bit
#   overkill too (having a time complexity O(|V|^3)).
# - This problem took over an hour to do. The odd cycles property threw me off,
#   making me think that I needed to keep track of node path lengths. Once I let
#   go of that idea, I realized that a greedy coloring approach would do the
#   trick.
def isBipartite(graph: list[list[int]]) -> bool:
    """
    Examples:
    >>> isBipartite([[1,2,3], [0,2], [0,1,3], [0,2]])
    False
    >>> isBipartite([[1, 3], [0, 2], [1, 3], [0, 2]])
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
