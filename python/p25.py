from collections import defaultdict, deque


def p2493(n: int, edges: list[list[int]]) -> int:
    """
    2493. Divide Nodes Into The Maximum Number of Groups https://leetcode.com/problems/divide-nodes-into-the-maximum-number-of-groups/

    Lessons learned:
    - This problem is a pretty straightforward extension of (785 Bipartite Graph
    Checking).
    - I spent a good 30-45 minutes not realizing that I was returning the minimum
    groups instead of maximum. Derp.
    - Checking only nodes with minimum degree <= node degree <= minimum degree + 1
    in a given partition yields a substantial savings (98 percentile
    performance), but I don't quite know if this is a general property or just a
    heuristic that helps on this particular test set. The intuition is that we
    can maximize graph diameter by starting at a leaf, leaves have degree 1, and
    so maybe looking for the smallest degree nodes leads us to outer boundary of
    a graph. Not sure.

    Examples:
    >>> p2493(6, [[1,2],[1,4],[1,5],[2,6],[2,3],[4,6]])
    4
    >>> p2493(3, [[1,2],[2,3],[3,1]])
    -1
    """
    # Convert from edge list to adjacency list
    graph = defaultdict(list)

    for u, v in edges:
        graph[u - 1].append(v - 1)
        graph[v - 1].append(u - 1)

    # Find connected components (store the lowest index member)
    seen: set[int] = set()
    partitions: dict[int, set[int]] = defaultdict(set)

    def dfs_connected_components(node: int, partition: int):
        if node not in seen:
            seen.add(node)
            partitions[partition].add(node)
            for neighbor_node in graph[node]:
                dfs_connected_components(neighbor_node, partition)

    for node in range(n):
        if node not in seen:
            dfs_connected_components(node, node)

    # Get a coloring for each connected component {partition: {node: color}}
    coloring = {}

    def bfs_coloring(root: int) -> bool:
        queue: deque[tuple[int, int]] = deque()
        queue.append((root, 0))
        while queue:
            node, color = queue.popleft()

            if node not in coloring:
                coloring[node] = color

                for neighbor_node in graph[node]:
                    if neighbor_node in coloring and (coloring[neighbor_node] - color - 1) % 2 == 1:
                        return False
                    if neighbor_node not in coloring:
                        queue.append((neighbor_node, color + 1))

        return True

    # Do BFS from every node, building a spanning tree, and looking for the maximum depth achieved
    result = 0
    max_coloring = -1
    for _, partition_nodes in partitions.items():
        for node in partition_nodes:
            if not bfs_coloring(node):
                return -1

            max_coloring = max(max_coloring, max(coloring.values()) + 1)
            coloring = {}
        result += max_coloring
        max_coloring = -1

    # A little degree checking heuristic that gives a big boost, but might not work in general.
    # result = 0
    # max_coloring = -1
    # for _, partition_nodes in partitions.items():
    #     min_degree = min(len(graph[node]) for node in partition_nodes)
    #     check_nodes = [node for node in partition_nodes if len(graph[node]) <= min_degree + 1]

    #     for node in check_nodes:
    #         if not bfs_coloring(node):
    #             return -1
    #         else:
    #             max_coloring = max(max_coloring, max(coloring.values()) + 1)
    #             coloring = defaultdict(dict)
    #     result += max_coloring
    #     max_coloring = -1

    return result
