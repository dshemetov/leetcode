# %% 24. Swap Nodes in Pairs https://leetcode.com/problems/swap-nodes-in-pairs/
from p2 import ListNode, list_to_listnode, listnode_to_list


def swapPairs(head: ListNode | None) -> ListNode | None:
    """
    Examples:
    >>> listnode_to_list(swapPairs(list_to_listnode([])))
    []
    >>> listnode_to_list(swapPairs(list_to_listnode([1])))
    [1]
    >>> listnode_to_list(swapPairs(list_to_listnode([1, 2])))
    [2, 1]
    >>> listnode_to_list(swapPairs(list_to_listnode([1, 2, 3])))
    [2, 1, 3]
    >>> listnode_to_list(swapPairs(list_to_listnode([1, 2, 3, 4])))
    [2, 1, 4, 3]
    >>> listnode_to_list(swapPairs(list_to_listnode([1, 2, 3, 4, 5])))
    [2, 1, 4, 3, 5]
    """
    if not head:
        return None
    if not head.next:
        return head

    pointer = head
    new_head = head.next

    # 1 2 3 4
    # 2 1 4 3
    while pointer and pointer.next:
        one = pointer
        two = pointer.next
        three = pointer.next.next
        four = pointer.next.next.next if three else None

        one.next = four if four else three
        two.next = one

        pointer = three

    return new_head
