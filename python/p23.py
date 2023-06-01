# %% 23. Merge k Sorted Lists https://leetcode.com/problems/merge-k-sorted-lists/
# Lessons learned:
# - Sometimes the hard problems aren't that hard.
from p2 import ListNode, list_to_listnode, listnode_to_list
from p21 import mergeTwoLists


def mergeKLists(lists: list[ListNode | None]) -> ListNode | None:
    """
    Examples:
    >>> listnode_to_list(mergeKLists([list_to_listnode([1, 4, 5]), list_to_listnode([1, 3, 4]), list_to_listnode([2, 6])]))
    [1, 1, 2, 3, 4, 4, 5, 6]
    >>> listnode_to_list(mergeKLists([]))
    []
    >>> listnode_to_list(mergeKLists([list_to_listnode([])]))
    []
    """
    head = pointer = ListNode()

    while any(x for x in lists):
        min_val = float("inf")
        min_idx = -1
        for i, x in enumerate(lists):
            if x and x.val < min_val:
                min_val = x.val
                min_idx = i

        pointer.next = lists[min_idx]
        lists[min_idx] = lists[min_idx].next
        pointer = pointer.next

    return head.next


def mergeKLists2(lists: list[ListNode | None]) -> ListNode | None:
    """
    Examples:
    >>> listnode_to_list(mergeKLists2([list_to_listnode([1, 4, 5]), list_to_listnode([1, 3, 4]), list_to_listnode([2, 6])]))
    [1, 1, 2, 3, 4, 4, 5, 6]
    >>> listnode_to_list(mergeKLists2([]))
    []
    >>> listnode_to_list(mergeKLists2([list_to_listnode([])]))
    []
    """
    if not lists:
        return None

    while len(lists) > 1:
        lists.append(mergeTwoLists(lists.pop(0), lists.pop(0)))

    return lists[0]
