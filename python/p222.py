# %% 222. Count Complete Tree Nodes https://leetcode.com/problems/count-complete-tree-nodes/


# Lessons learned:
# - A complete binary tree is a binary tree in which every level, except
# - possibly the last, is completely filled,
#   and all nodes in the last level are as far left as possible.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def make_binary_tree(lst: list[int]) -> TreeNode:
    """Fills a binary tree from left to right."""
    if not lst:
        return None
    root = TreeNode(lst[0])
    queue = [root]
    i = 1
    while i < len(lst):
        node = queue.pop(0)
        if lst[i] is not None:
            node.left = TreeNode(lst[i])
            queue.append(node.left)
        i += 1
        if i < len(lst) and lst[i] is not None:
            node.right = TreeNode(lst[i])
            queue.append(node.right)
        i += 1
    return root


def countNodes(root: TreeNode | None) -> int:
    """
    Examples:
    >>> countNodes(make_binary_tree([1,2,3,4,5,6]))
    6
    >>> countNodes(make_binary_tree([1,2,3,4,5,6,None]))
    6
    >>> countNodes(make_binary_tree([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]))
    15
    >>> countNodes(make_binary_tree([1,2,3,4,5,6,7,8,9,10,11,12,None,None,None]))
    12
    """
    if not root:
        return 0

    height = -1
    node = root
    while node:
        height += 1
        node = node.left

    if height == 0:
        return 1

    def is_node_in_tree(root: TreeNode, i: int) -> bool:
        node = root
        for c in format(i, f"0{height}b"):
            node = node.left if c == "0" else node.right
        return bool(node)

    lo, hi = 0, 2 ** (height) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        node_in_tree = is_node_in_tree(root, mid)
        if node_in_tree:
            lo = mid + 1
        else:
            hi = mid - 1

    return 2 ** (height) + lo - 1
