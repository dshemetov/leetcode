# %% 1721. Swapping Nodes in a Linked List https://leetcode.com/problems/swapping-nodes-in-a-linked-list/


# Lessons learned:
# - Two pointers allows you to do this in one pass.
from p2 import ListNode, list_to_listnode, listnode_to_list


def swapNodes(head: ListNode, k: int) -> ListNode:
    """
    Examples:
    >>> listnode_to_list(swapNodes(list_to_listnode([1,2,3,4,5]), 2))
    [1, 4, 3, 2, 5]
    >>> listnode_to_list(swapNodes(list_to_listnode([7,9,6,6,7,8,3,0,9,5]), 5))
    [7, 9, 6, 6, 8, 7, 3, 0, 9, 5]
    """
    p1 = head
    for _ in range(k - 1):
        p1 = p1.next
    node1 = p1
    p2 = head
    while p1.next:
        p1 = p1.next
        p2 = p2.next
    node2 = p2
    node1.val, node2.val = node2.val, node1.val
    return head
