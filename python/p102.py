# %% 102. Binary Tree Level Order Traversal https://leetcode.com/problems/binary-tree-level-order-traversal/
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def levelOrder(root: TreeNode | None) -> list[list[int]]:
    """
    Examples:
    >>> levelOrder(TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7))))
    [[3], [9, 20], [15, 7]]
    >>> levelOrder(TreeNode(1))
    [[1]]
    >>> levelOrder(None)
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
