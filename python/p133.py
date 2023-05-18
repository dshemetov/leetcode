# %% 133. Clone Graph https://leetcode.com/problems/clone-graph/
from collections import deque


class Node:
    def __init__(self, val: int = 0, neighbors: list["Node"] | None = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def adjacency_list_to_node_graph(adjacency_list: list[list[int]]) -> "Node":
    """Build a node-based graph from an adjacency list.

    Examples:
    >>> node_graph_to_adjacency_list(adjacency_list_to_node_graph([[1, 2], [1, 4], [2, 3], [3, 4]]))
    [[1, 2], [1, 4], [2, 3], [3, 4]]
    """
    if adjacency_list == [[]]:
        return Node(1)

    node_index = {}
    for x, y in adjacency_list:
        if (xnode := node_index.get(x)) is None:
            xnode = Node(x)
            node_index[x] = xnode
        if (ynode := node_index.get(y)) is None:
            ynode = Node(y)
            node_index[y] = ynode

        xnode.neighbors.append(ynode)
        ynode.neighbors.append(xnode)

    return node_index[1]


def node_graph_to_adjacency_list(node: "Node") -> "Node":
    """Traverse through a graph and build an adjacency list.

    Examples:
    >>> node_graph_to_adjacency_list(adjacency_list_to_node_graph([[1, 2], [1, 4], [2, 3], [3, 4]]))
    [[1, 2], [1, 4], [2, 3], [3, 4]]
    """
    adjacency_list = set()
    visited = set()
    node_queue = deque([node])

    while node_queue:
        node = node_queue.popleft()
        visited.add(node.val)

        for neighbor in node.neighbors:
            adjacency_list.add(tuple(sorted([node.val, neighbor.val])))

            if neighbor.val not in visited:
                node_queue.append(neighbor)

    return sorted([list(e) for e in adjacency_list], key=lambda x: (x[0], x[1]))


def cloneGraph(node: "Node") -> "Node":
    """
    Examples:
    >>> cloneGraph(None)
    >>> node_graph_to_adjacency_list(cloneGraph(adjacency_list_to_node_graph([[1, 2], [1, 4], [2, 3], [3, 4]])))
    [[1, 2], [1, 4], [2, 3], [3, 4]]
    """
    if node is None:
        return None

    node_queue = deque([node])
    clone_index = {node.val: Node(node.val)}
    while node_queue:
        cur_node = node_queue.popleft()
        cur_clone = clone_index[cur_node.val]

        for neighbor in cur_node.neighbors:
            if neighbor.val not in clone_index:
                clone_index[neighbor.val] = Node(neighbor.val)
                node_queue.append(neighbor)

            cur_clone.neighbors.append(clone_index[neighbor.val])

    return clone_index[1]
