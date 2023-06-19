# %% 2. Add Two Numbers https://leetcode.com/problems/add-two-numbers/
class ListNode:
    def __init__(self, val=0, next=None):  # pylint: disable=redefined-builtin
        self.val = val
        self.next = next


def list_to_listnode(lst: list[int]) -> ListNode | None:
    if not lst:
        return None

    original_head = head = ListNode(lst[0])
    for x in lst[1:]:
        head.next = ListNode(x)
        head = head.next

    return original_head


def listnode_to_list(head: ListNode) -> list[int]:
    """
    Examples:
    >>> listnode_to_list(list_to_listnode([1, 2, 3, 4, 5]))
    [1, 2, 3, 4, 5]
    """
    lst = []
    while head:
        lst.append(head.val)
        head = head.next

    return lst


def add_two_numbers(l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
    """
    Examples:
    >>> list_to_int(add_two_numbers(int_to_list(0), int_to_list(15)))
    15
    >>> list_to_int(add_two_numbers(int_to_list(12), int_to_list(15)))
    27
    >>> list_to_int(add_two_numbers(int_to_list(12), int_to_list(153)))
    165
    """
    first_node = ListNode(0)
    cur_node = first_node
    carry_bit = 0
    while l1 or l2 or carry_bit > 0:
        x = l1.val if l1 else 0
        y = l2.val if l2 else 0

        num = x + y + carry_bit
        carry_bit = num // 10

        new_node = ListNode(num % 10)
        cur_node.next = new_node
        cur_node = new_node

        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None

    return first_node.next


def int_to_list(i: int) -> ListNode:
    num_list = None
    for x in str(i):
        num_list = ListNode(val=int(x), next=num_list)
    return num_list


def list_to_int(lst: ListNode) -> int:
    """
    Examples:
    >>> list_to_int(int_to_list(0))
    0
    >>> list_to_int(int_to_list(2))
    2
    >>> list_to_int(int_to_list(12))
    12
    >>> list_to_int(int_to_list(15))
    15
    >>> list_to_int(int_to_list(255))
    255
    """
    num = 0
    digit = 0
    while lst:
        num += lst.val * 10**digit
        digit += 1
        lst = lst.next
    return num
