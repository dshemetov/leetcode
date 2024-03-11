from bisect import insort
from collections import Counter


def p212(board: list[list[str]], words: list[str]) -> list[str]:
    """
    212. Word Search II https://leetcode.com/problems/word-search-ii/

    Examples:
    >>> set(p212([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], ["oath","pea","eat","rain"])) == set(["eat", "oath"])
    True
    >>> p212([["a","b"],["c","d"]], ["abcb"])
    []
    >>> board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
    >>> words = ["oath","pea","eat","rain", "oat", "oatht", "naaoetaerkhi", "naaoetaerkhii"]
    >>> set(p212(board, words)) == set(["eat", "oath", "oat", "naaoetaerkhi"])
    True
    """

    class Trie:
        def __init__(self):
            self.root = {}

        def insert(self, word: str) -> None:
            node = self.root
            for char in word:
                node = node.setdefault(char, {})
            node["#"] = "#"

        def remove(self, word: str) -> None:
            node = self.root
            path = []
            for char in word:
                path.append((node, char))
                node = node[char]
            node.pop("#")
            for node, char in reversed(path):
                if not node[char]:
                    node.pop(char)
                else:
                    break

    def dfs(
        i: int,
        j: int,
        node: dict,
        path: str,
        board: list[list[str]],
        found_words: set[str],
    ) -> None:
        if node.get("#"):
            found_words.add(path)
            trie.remove(path)

        board[i][j] = "$"

        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ni, nj = (i + di, j + dj)
            if (
                0 <= ni < len(board)
                and 0 <= nj < len(board[0])
                and board[ni][nj] in node
                and len(path) < 12
            ):
                dfs(
                    ni,
                    nj,
                    node[board[ni][nj]],
                    path + board[ni][nj],
                    board,
                    found_words,
                )

        board[i][j] = path[-1]

    def filter_words(words: list[str]) -> list[str]:
        board_chars = set()
        for row in board:
            board_chars |= set(row)
        return [word for word in words if set(word) <= board_chars]

    words = filter_words(words)

    trie = Trie()
    for word in words:
        trie.insert(word)

    n, m = len(board), len(board[0])
    found_words = set()
    for i in range(n):
        for j in range(m):
            if board[i][j] in trie.root:
                dfs(i, j, trie.root[board[i][j]], board[i][j], board, found_words)

    return list(found_words)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def make_binary_tree(lst: list[int]) -> TreeNode:
    """Fills a binary tree from left to right."""
    if not lst:
        return None
    root = TreeNode(lst[0])
    queue = [root]
    i = 1
    while i < len(lst):
        node = queue.pop(0)
        if lst[i] is not None:
            node.left = TreeNode(lst[i])
            queue.append(node.left)
        i += 1
        if i < len(lst) and lst[i] is not None:
            node.right = TreeNode(lst[i])
            queue.append(node.right)
        i += 1
    return root


def p222(root: TreeNode | None) -> int:
    """
    222. Count Complete Tree Nodes https://leetcode.com/problems/count-complete-tree-nodes/

    Lessons learned:
    - A complete binary tree is a binary tree in which every level is completely
    filled, except for the last where the nodes must be as far left as possible.

    Examples:
    >>> p222(make_binary_tree([1,2,3,4,5,6]))
    6
    >>> p222(make_binary_tree([1,2,3,4,5,6,None]))
    6
    >>> p222(make_binary_tree([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]))
    15
    >>> p222(make_binary_tree([1,2,3,4,5,6,7,8,9,10,11,12,None,None,None]))
    12
    """
    if not root:
        return 0

    height = -1
    node = root
    while node:
        height += 1
        node = node.left

    if height == 0:
        return 1

    def is_node_in_tree(root: TreeNode, i: int) -> bool:
        node = root
        for c in format(i, f"0{height}b"):
            node = node.left if c == "0" else node.right
        return bool(node)

    lo, hi = 0, 2 ** (height) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        node_in_tree = is_node_in_tree(root, mid)
        if node_in_tree:
            lo = mid + 1
        else:
            hi = mid - 1

    return 2 ** (height) + lo - 1


def p223(ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
    """
    223. Rectangle Area https://leetcode.com/problems/rectangle-area/
    """
    A1 = (ax2 - ax1) * (ay2 - ay1)
    A2 = (bx2 - bx1) * (by2 - by1)
    I = max(min(ax2, bx2) - max(ax1, bx1), 0) * max(min(ay2, by2) - max(ay1, by1), 0)
    return A1 + A2 - I


def p242(s: str, t: str) -> bool:
    """
    242. Valid Anagram https://leetcode.com/problems/valid-anagram/
    """
    return Counter(s) == Counter(t)


def p258(num: int) -> int:
    """
    258. Add Digits https://leetcode.com/problems/add-digits/

    Lessons learned:
    - Turns out this can be solved with modular arithmetic because 10 ** n == 1 mod 9

    Examples:
    >>> p258(38)
    2
    >>> p258(0)
    0
    """
    if num == 0:
        return num
    if num % 9 == 0:
        return 9
    return num % 9


def p263(n: int) -> bool:
    """
    263. Ugly Number https://leetcode.com/problems/ugly-number/
    """
    if n < 1:
        return False
    while n % 2 == 0:
        n /= 2
    while n % 3 == 0:
        n /= 3
    while n % 5 == 0:
        n /= 5
    return n == 1


class p295:
    """
    295. Find Median From Data Stream https://leetcode.com/problems/find-median-from-data-stream/

    Examples:
    >>> mf = p295()
    >>> mf.addNum(1)
    >>> mf.addNum(2)
    >>> mf.findMedian()
    1.5
    >>> mf.addNum(3)
    >>> mf.findMedian()
    2.0
    >>> mf = p295()
    >>> mf.addNum(1)
    >>> mf.addNum(2)
    >>> mf.addNum(3)
    >>> mf.addNum(4)
    >>> mf.addNum(5)
    >>> mf.addNum(6)
    >>> mf.addNum(7)
    >>> mf.findMedian()
    4.0
    >>> mf = p295()
    >>> mf.addNum(-1)
    >>> mf.addNum(-2)
    >>> mf.addNum(-3)
    >>> mf.heap
    [-3, -2, -1]
    >>> mf.findMedian()
    -2.0
    """

    def __init__(self):
        self.heap = []

    def addNum(self, num: int) -> None:
        insort(self.heap, num)

    def findMedian(self) -> float:
        if len(self.heap) % 2 == 1:
            return float(self.heap[len(self.heap) // 2])
        return (self.heap[len(self.heap) // 2] + self.heap[len(self.heap) // 2 - 1]) / 2
