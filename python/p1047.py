# %% 1047. Remove All Adjacent Duplicates in String https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/
def remove_duplicates(s: str) -> str:
    """
    Examples:
    >>> remove_duplicates("abbaca")
    'ca'
    >>> remove_duplicates("aaaaaaaa")
    ''
    """
    stack = []
    for c in s:
        if stack and c == stack[-1]:
            stack.pop()
        else:
            stack.append(c)
    return "".join(stack)
