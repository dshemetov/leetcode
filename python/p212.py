# %% 212. Word Search II https://leetcode.com/problems/word-search-ii/
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    """
    Examples:
    >>> set(findWords([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], ["oath","pea","eat","rain"])) == set(["eat", "oath"])
    True
    >>> findWords([["a","b"],["c","d"]], ["abcb"])
    []
    >>> board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
    >>> words = ["oath","pea","eat","rain", "oat", "oatht", "naaoetaerkhi", "naaoetaerkhii"]
    >>> set(findWords(board, words)) == set(["eat", "oath", "oat", "naaoetaerkhi"])
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
