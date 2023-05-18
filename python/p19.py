# %% 19. Remove Nth Node From End of List https://leetcode.com/problems/remove-nth-node-from-end-of-list/
from p2 import ListNode, list_to_listnode, listnode_to_list


def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
    """
    Examples:
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2, 3, 4, 5]), 1))
    [1, 2, 3, 4]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2, 3, 4, 5]), 2))
    [1, 2, 3, 5]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2, 3, 4, 5]), 3))
    [1, 2, 4, 5]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2, 3, 4, 5]), 4))
    [1, 3, 4, 5]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2, 3, 4, 5]), 5))
    [2, 3, 4, 5]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1]), 1))
    []
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2]), 1))
    [1]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2]), 2))
    [2]
    """
    sz = 0
    node = head
    while node:
        node = node.next
        sz += 1

    if sz == 1:
        return None

    if sz == n:
        return head.next

    node = head
    for _ in range(sz - n - 1):
        node = node.next

    node.next = node.next.next

    return head
