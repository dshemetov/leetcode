# %% 1544. Make The String Great https://leetcode.com/problems/make-the-string-great/
def makeGood(s: str) -> str:
    """
    Examples:
    >>> makeGood("leEeetcode")
    'leetcode'
    >>> makeGood("abBAcC")
    ''
    >>> makeGood("s")
    's'
    """
    stack = []
    for c in s:
        if stack and stack[-1].lower() == c.lower() and stack[-1] != c:
            stack.pop()
        else:
            stack.append(c)
    return "".join(stack)
