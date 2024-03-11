from typing import Optional

from p01 import ListNode, listnode_to_list


def is_similar(s1: str, s2: str) -> bool:
    """
    Examples:
    >>> is_similar("abc", "abc")
    True
    >>> is_similar("abc", "acb")
    True
    >>> is_similar("abc", "abcd")
    False
    >>> is_similar("abc", "abd")
    False
    """
    if len(s1) != len(s2):
        return False
    diff_chars = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return diff_chars in {0, 2}


def p839(strs: list[str]) -> int:
    """
    839. Similar String Groups https://leetcode.com/problems/similar-string-groups/

    Examples:
    >>> p839(["tars","rats","arts","star"])
    2
    >>> p839(["omv","ovm"])
    1
    >>> p839(["a"])
    1
    >>> p839(["abc","abc"])
    1
    >>> p839(["abc","acb","abc","acb","abc","acb"])
    1
    """
    n = len(strs)
    parent: dict[int, int] = dict({i: i for i in range(n)})

    def find(x: str) -> str:
        y = x
        while True:
            if y != parent[y]:
                y = parent[y]
                continue
            break
        parent[x] = y
        return parent[x]

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if is_similar(strs[i], strs[j]):
                union(i, j)

    return len({find(i) for i in range(n)})


def p876(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    876. Middle of the Linked List https://leetcode.com/problems/middle-of-the-linked-list

    Examples:
    >>> listnode_to_list(p876(ListNode.from_list([1, 2, 3, 4, 5])))
    [3, 4, 5]
    >>> listnode_to_list(p876(ListNode.from_list([1, 2, 3, 4, 5, 6])))
    [4, 5, 6]
    """
    if not head or not head.next:
        return head
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def p899(s: str, k: int) -> str:
    """
    899. Orderly Queue https://leetcode.com/problems/orderly-queue/

    Lessons learned:
    - This problem is such a troll. At first I thought I found a totally
    ridiculous Copilot suggestion, but then I realized that the solution was
    actually dead simple - you can use the rightmost character as a register and
    rotate the string until the correct insertion point.

    Examples:
    >>> p899("cba", 1)
    'acb'
    >>> p899("baaca", 3)
    'aaabc'
    >>> p899("baaca", 1)
    'aacab'
    >>> p899("baaca", 2)
    'aaabc'
    >>> p899("baaca", 4)
    'aaabc'
    >>> p899("badaca", 2)
    'aaabcd'
    >>> p899("badacadeff", 3)
    'aaabcddeff'
    """
    if k == 1:
        return min(s[i:] + s[:i] for i in range(len(s)))

    return "".join(sorted(s))
