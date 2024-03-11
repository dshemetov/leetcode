from collections import Counter, defaultdict, deque
from bisect import bisect_left

import numpy as np


def p316(s: str) -> str:
    """
    316. Remove Duplicate Letters https://leetcode.com/problems/remove-duplicate-letters/
    1081. https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/

    Lessons learned:
    - In this one, the solution heuristic can be established with a few examples.
    The key is that we can greedily remove left-most duplicated letters that are
    larger than the next letter. For example, if we have cbxxx and we can remove
    c or another letter, then we will have bxxx < cbxx.

    Examples:
    >>> p316("bcabc")
    'abc'
    >>> p316("cbacdcbc")
    'acdb'
    >>> p316("bbcaac")
    'bac'
    >>> p316("bcba")
    'bca'
    """
    letter_counts = Counter(s)
    stack = []
    for c in s:
        letter_counts[c] -= 1
        if c in stack:
            continue
        while stack and c < stack[-1] and letter_counts[stack[-1]] > 0:
            stack.pop()
        stack.append(c)
    return "".join(stack)


def p319(n: int) -> int:
    """
    319. Bulb Switcher https://leetcode.com/problems/bulb-switcher/

    Lessons learned:
    - Testing the array at n=50, I saw that only square numbers remained. From
    there it was easy to prove that square numbers are the only ones with an odd
    number of factors. So this problem is just counting the number of perfect
    squares <= n.

    Examples:
    >>> p319(3)
    1
    >>> p319(0)
    0
    >>> p319(1)
    1
    >>> p319(5)
    2
    """
    arr = np.zeros(n, dtype=int)
    for i in range(1, n + 1):
        for j in range(0, n):
            if (j + 1) % i == 0:
                arr[j] = 1 if arr[j] == 0 else 0
    return sum(arr)


def p319_2(n: int) -> int:
    """
    Examples:
    >>> p319_2(3)
    1
    >>> p319_2(0)
    0
    >>> p319_2(1)
    1
    >>> p319_2(5)
    2
    """
    return int(np.sqrt(n))


def p345(s: str) -> str:
    """
    345. Reverse Vowels of a String https://leetcode.com/problems/reverse-vowels-of-a-string/

    Examples:
    >>> p345("hello")
    'holle'
    >>> p345("leetcode")
    'leotcede'
    """
    if len(s) == 1:
        return s

    hi = len(s) - 1
    s_ = []
    for c in s:
        if c in "aeiouAEIOU":
            while s[hi] not in "aeiouAEIOU":
                hi -= 1
            s_.append(s[hi])
            hi -= 1
        else:
            s_.append(c)

    return "".join(s_)


def p347(nums: list[int], k: int) -> list[int]:
    """
    347. Top K Frequent Elements https://leetcode.com/problems/top-k-frequent-elements/

    Examples:
    >>> p347([1,1,1,2,2,3], 2)
    [1, 2]
    >>> p347([1], 1)
    [1]
    """
    c = Counter(nums)
    return [num for num, _ in c.most_common(k)]


__pick__ = 6


def guess(num: int) -> int:
    if num == __pick__:
        return 0
    if num > __pick__:
        return -1
    return 1


def p374(n: int) -> int:
    """
    374. Guess Number Higher or Lower https://leetcode.com/problems/guess-number-higher-or-lower/

    Lessons learned:
    - bisect_left has a 'key' argument as of 3.10.

    Examples:
    >>> p374(10)
    6
    """
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        out = guess(mid)
        if out == 1:
            lo = mid + 1
        elif out == -1:
            hi = mid - 1
        else:
            return mid

    return lo


def p374_2(n: int) -> int:
    """
    Examples:
    >>> p374_2(10)
    6
    """
    return bisect_left(range(0, n), 0, lo=0, hi=n, key=lambda x: -guess(x))


def p399(equations: list[list[str]], values: list[float], queries: list[list[str]]) -> list[float]:
    """
    399. Evaluate Division https://leetcode.com/problems/evaluate-division/

    Examples:
    >>> p399([["a","b"],["b","c"]], [2.0,3.0], [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]])
    [6.0, 0.5, -1.0, 1.0, -1.0]
    >>> p399([["a","b"],["b","c"],["bc","cd"]], [1.5,2.5,5.0], [["a","c"],["c","b"],["bc","cd"],["cd","bc"]])
    [3.75, 0.4, 5.0, 0.2]
    >>> p399([["a","b"]], [0.5], [["a","b"],["b","a"],["a","c"],["x","y"]])
    [0.5, 2.0, -1.0, -1.0]
    """
    graph: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), v in zip(equations, values):
        graph[a][b] = v
        graph[b][a] = 1 / v

    def dfs(a: str, b: str) -> float:
        if a not in graph or b not in graph:
            return -1.0
        unexplored = deque([(a, 1.0)])
        visited = set()
        while unexplored:
            node, cost = unexplored.pop()
            visited.add(node)
            if node == b:
                return cost
            for child in graph[node]:
                if child not in visited:
                    unexplored.append((child, cost * graph[node][child]))
        return -1.0

    return [dfs(a, b) for a, b in queries]
