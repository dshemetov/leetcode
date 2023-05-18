# %% 2130. Maximum Twin Sum of a Linked List https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/
from p2 import ListNode, list_to_listnode, listnode_to_list


# Lessons learned:
# - Finding the midpoint of a linked list can be done with two pointers.
#   Reversing a linked list is pretty easy. These steps above can be done in one
#   pass.
def maxTwinSum(head: ListNode | None) -> int:
    """
    Examples:
    >>> maxTwinSum(list_to_listnode([5,4,2,1]))
    6
    >>> maxTwinSum(list_to_listnode([4,2,2,3]))
    7
    >>> maxTwinSum(list_to_listnode([1,100000]))
    100001
    """
    if head is None:
        return 0

    # Find the midpoint
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse the second half
    prev = None
    while slow:
        slow.next, prev, slow = prev, slow, slow.next

    # Find the maximum sum
    m = 0
    while prev:
        m = max(m, prev.val + head.val)
        prev = prev.next
        head = head.next

    return m
