from bisect import insort


def p1035(nums1: list[int], nums2: list[int]) -> int:
    """
    1035. Uncrossed Lines https://leetcode.com/problems/uncrossed-lines/

    Lessons learned:
    - The solution is identical to (1143 Longest Common Subsequence).

    Examples:
    >>> p1035([1,4,2], [1,2,4])
    2
    >>> p1035([2,5,1,2,5], [10,5,2,1,5,2])
    3
    >>> p1035([1,3,7,1,7,5], [1,9,2,5,1])
    2
    """
    dp_ = [[0 for _ in range(len(nums2) + 1)] for _ in range(len(nums1) + 1)]

    for i in range(1, len(nums1) + 1):
        for j in range(1, len(nums2) + 1):
            if nums1[i - 1] == nums2[j - 1]:
                dp_[i][j] = 1 + dp_[i - 1][j - 1]
            else:
                dp_[i][j] = max(dp_[i - 1][j], dp_[i][j - 1])

    return dp_[-1][-1]


def p1046(stones: list[int]) -> int:
    """
    1046. Last Stone Weight https://leetcode.com/problems/last-stone-weight/

    Examples:
    >>> p1046([2,7,4,1,8,1])
    1
    >>> p1046([1,3])
    2
    >>> p1046([1])
    1
    """
    sorted_stones = sorted(stones)
    while len(sorted_stones) > 1:
        a, b = sorted_stones.pop(), sorted_stones.pop()
        if a != b:
            insort(sorted_stones, a - b)
    return sorted_stones[0] if sorted_stones else 0


def p1047(s: str) -> str:
    """
    1047. Remove All Adjacent Duplicates in String https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/

    Examples:
    >>> p1047("abbaca")
    'ca'
    >>> p1047("aaaaaaaa")
    ''
    """
    stack = []
    for c in s:
        if stack and c == stack[-1]:
            stack.pop()
        else:
            stack.append(c)
    return "".join(stack)
