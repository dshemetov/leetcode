# %% 839. Similar String Groups https://leetcode.com/problems/similar-string-groups/
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


def numSimilarGroups(strs: list[str]) -> int:
    """
    Examples:
    >>> numSimilarGroups(["tars","rats","arts","star"])
    2
    >>> numSimilarGroups(["omv","ovm"])
    1
    >>> numSimilarGroups(["a"])
    1
    >>> numSimilarGroups(["abc","abc"])
    1
    >>> numSimilarGroups(["abc","acb","abc","acb","abc","acb"])
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
