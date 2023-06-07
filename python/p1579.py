# %% 1579. Remove Max Number of Edges to Keep Graph Fully Traversable https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/


# Lessons learned:
# - We can build a spanning tree greedily by adding edges when they don't create
#   a cycle. We can detect when an edge would create a cycle, by using a
#   disjoint set. Counting these edges gives us the number removable edges. This
#   problem adds a minor complication by having three types of edges. This
#   complication can be dealth with by keeping track of two graphs. Since
#   sometimes one edge of type 3 can make two edges of type 1 and 2 obsolete, we
#   prioritize adding edges of type 3 first.
# - A spanning tree always has the minimum number of edges to connect all nodes,
#   which is V - 1 for a graph with V nodes
def maxNumEdgesToRemove(n: int, edges: list[list[int]]) -> int:
    """
    Examples:
    >>> maxNumEdgesToRemove(4, [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]])
    2
    >>> maxNumEdgesToRemove(4, [[3,1,2],[3,2,3],[1,1,4],[2,1,4]])
    0
    >>> maxNumEdgesToRemove(4, [[3,2,3],[1,1,2],[2,3,4]])
    -1
    >>> maxNumEdgesToRemove(2, [[1,1,2],[2,1,2],[3,1,2]])
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
