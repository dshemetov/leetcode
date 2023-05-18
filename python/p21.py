# %% 21. Merge Two Sorted Lists https://leetcode.com/problems/merge-two-sorted-lists/
from p2 import ListNode, list_to_listnode, listnode_to_list


def mergeTwoLists(list1: ListNode | None, list2: ListNode | None) -> ListNode | None:
    """
    Examples:
    >>> listnode_to_list(mergeTwoLists(list_to_listnode([1, 2, 4]), list_to_listnode([1, 3, 4])))
    [1, 1, 2, 3, 4, 4]
    >>> listnode_to_list(mergeTwoLists(list_to_listnode([]), list_to_listnode([])))
    []
    >>> listnode_to_list(mergeTwoLists(list_to_listnode([]), list_to_listnode([0])))
    [0]
    """
    head = pointer = ListNode()

    while list1 and list2:
        if list1.val < list2.val:
            pointer.next = list1
            list1 = list1.next
        else:
            pointer.next = list2
            list2 = list2.next
        pointer = pointer.next

    if list1:
        pointer.next = list1

    if list2:
        pointer.next = list2

    return head.next
