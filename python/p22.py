from collections import Counter

from p01 import ListNode, list_to_listnode, listnode_to_list


def p2130(head: ListNode | None) -> int:
    """
    2130. Maximum Twin Sum of a Linked List https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/

    Lessons learned:
    - Finding the midpoint of a linked list can be done with two pointers.
      Reversing a linked list is pretty easy. These steps above can be done in one
      pass.

    Examples:
    >>> p2130(list_to_listnode([5,4,2,1]))
    6
    >>> p2130(list_to_listnode([4,2,2,3]))
    7
    >>> p2130(list_to_listnode([1,100000]))
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


def p2131(words: list[str]) -> int:
    """
    2131. Longest Palindrome by Concatenating Two Letter Words https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/

    Examples:
    >>> p2131(["ab","ba","aa","bb","cc"])
    6
    >>> p2131(["ab","ba","cc","ab","ba","cc"])
    12
    >>> p2131(["aa","ba"])
    2
    >>> p2131(["ba", "ce"])
    0
    >>> p2131(["lc","cl","gg"])
    6
    >>> p2131(["ab","ty","yt","lc","cl","ab"])
    8
    >>> p2131(["cc","ll","xx"])
    2
    """
    d: Counter[str] = Counter()

    for word in words:
        d[word] += 1

    total = 0
    extra_double = False
    for key in d:
        if key[0] == key[1]:
            total += d[key] // 2 * 4
            if d[key] % 2 == 1:
                extra_double = True
        elif key == "".join(sorted(key)):
            total += min(d[key], d[key[::-1]]) * 4

    if extra_double:
        total += 2

    return total
