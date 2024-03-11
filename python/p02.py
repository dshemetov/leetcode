from array import array
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def p102(root: TreeNode | None) -> list[list[int]]:
    """
    102. Binary Tree Level Order Traversal https://leetcode.com/problems/binary-tree-level-order-traversal/

    Examples:
    >>> p102(TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7))))
    [[3], [9, 20], [15, 7]]
    >>> p102(TreeNode(1))
    [[1]]
    >>> p102(None)
    []
    """
    if not root:
        return []

    queue = [root]
    result = []
    while queue:
        level = []
        new_queue = []
        for x in queue:
            level.append(x.val)
            if x.left:
                new_queue.append(x.left)
            if x.right:
                new_queue.append(x.right)
        result.append(level)
        queue = new_queue
    return result


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


def p133(node: "Node") -> "Node":
    """
    133. Clone Graph https://leetcode.com/problems/clone-graph/

    Examples:
    >>> p133(None)
    >>> node_graph_to_adjacency_list(p133(adjacency_list_to_node_graph([[1, 2], [1, 4], [2, 3], [3, 4]])))
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


def p151(s: str) -> str:
    """
    151. Reverse Words In A String https://leetcode.com/problems/reverse-words-in-a-string/

    Lesson learned:
    - Python string built-ins are fast.
    - The follow-up asks: can you use only O(1) extra space? Here is the trick:
    first reverse the whole string and then reverse each word. Reversing each
    requires keeping track of two pointers: the start of a word and the end of a
    word (terminated by a space).

    Examples:
    >>> p151("the sky is blue")
    'blue is sky the'
    >>> p151("  hello world!  ")
    'world! hello'
    >>> p151("a good   example")
    'example good a'
    >>> p151("  Bob    Loves  Alice   ")
    'Alice Loves Bob'
    """
    return " ".join(s.split()[::-1])


def p151_2(s: str) -> str:
    """
    Examples:
    >>> p151_2("the sky is blue")
    'blue is sky the'
    >>> p151_2("  hello world!  ")
    'world! hello'
    >>> p151_2("a good   example")
    'example good a'
    >>> p151_2("  Bob    Loves  Alice   ")
    'Alice Loves Bob'
    """
    a = array("u", [])
    a.fromunicode(s.strip())
    a.reverse()

    # Reverse each word
    n = len(a)
    lo = 0
    for i in range(n):
        if a[i] == " ":
            hi = i - 1
            while lo < hi:
                a[lo], a[hi] = a[hi], a[lo]
                lo += 1
                hi -= 1
            lo = i + 1

    hi = n - 1
    while lo < hi:
        a[lo], a[hi] = a[hi], a[lo]
        lo += 1
        hi -= 1

    # Contract spaces in the string
    lo, space = 0, 0
    for i in range(n):
        space = space + 1 if a[i] == " " else 0
        if space <= 1:
            a[lo] = a[i]
            lo += 1

    return "".join(a[:lo])


def p167(numbers: list[int], target: int) -> list[int]:
    """
    167. Two Sum II - Input Array Is Sorted https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/

    Examples:
    >>> p167([2,7,11,15], 9)
    [1, 2]
    >>> p167([2,3,4], 6)
    [1, 3]
    >>> p167([-1,0], -1)
    [1, 2]
    """
    lo, hi = 0, len(numbers) - 1
    while lo < hi:
        s = numbers[lo] + numbers[hi]
        if s < target:
            lo += 1
        elif s > target:
            hi -= 1
        else:
            return [lo + 1, hi + 1]
    return 0


def p200(grid: list[list[str]]) -> int:
    """
    200. Number of Islands https://leetcode.com/problems/number-of-islands/
    """
    n, m = len(grid), len(grid[0])
    visited = set()

    def dfs(i: int, j: int):
        unexplored = {(i, j)}
        while unexplored:
            i_, j_ = unexplored.pop()
            visited.add((i_, j_))

            for i__, j__ in [(i_ + 1, j_), (i_ - 1, j_), (i_, j_ + 1), (i_, j_ - 1)]:
                if (
                    0 <= i__ < n
                    and 0 <= j__ < m
                    and (i__, j__) not in visited
                    and grid[i__][j__] == "1"
                ):
                    unexplored.add((i__, j__))

    islands = 0
    for i in range(n):
        for j in range(m):
            if (i, j) not in visited and grid[i][j] == "1":
                dfs(i, j)
                islands += 1

    return islands
