# %% 990. Satisfiability of Equality Equations https://leetcode.com/problems/satisfiability-of-equality-equations/


# Lessons learned:
# - This was clearly a graph problem underneath, where you need to find the
#   connected components given by the equality statements
# - Efficiently calculating the connected components was hard for me though, so
#   learning about the disjoint set data structure was key (also referred to as
#   union find):
#   https://cp-algorithms.com/data_structures/disjoint_set_union.html
def equationsPossible(equations: list[str]) -> bool:
    """
    Examples:
    >>> assert equationsPossible(["a==b", "b!=a"]) is False
    >>> assert equationsPossible(["x==y", "z==w", "y==z", "a==b", "d==e", "f==g", "e==f", "w==x", "c==d", "b==d", "g!=x"]) is True
    >>> assert equationsPossible(["x==y", "z==w", "y==z", "a==b", "d==e", "f==g", "e==f", "w==x", "c==d", "b==d", "g!=x", "a==z"]) is False
    >>> assert equationsPossible(["x==a", "w==b", "z==c", "a==b", "b==c", "c!=x"]) is False
    >>> assert equationsPossible(["a==b", "c==e", "b==c", "a!=e"]) is False
    >>> assert equationsPossible(["a==b", "e==c", "c==b", "a!=e"]) is False
    >>> assert equationsPossible(["a==b", "e==c", "c==b", "a!=e"]) is False
    >>> assert equationsPossible(["a==b", "e==c", "b==c", "a!=e"]) is False
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while True:
            if parent[x] == x:
                return x
            parent[x] = parent[parent[x]]
            x = parent[x]

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for x, eq, _, y in equations:
        if eq == "=":
            parent.setdefault(x, x)
            parent.setdefault(y, y)
            union(x, y)

    for x, eq, _, y in equations:
        if eq == "!":
            if x == y:
                return False
            if find(x) == find(y):
                return False
    return True
