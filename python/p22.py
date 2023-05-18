# %% 22. Generate Parentheses https://leetcode.com/problems/generate-parentheses/
def generate_parenthesis(n: int) -> list[str]:
    """
    Examples:
    >>> generate_parenthesis(3)
    ['((()))', '(()())', '(())()', '()(())', '()()()']
    >>> generate_parenthesis(1)
    ['()']
    """
    if n == 1:
        return ["()"]

    res: list[str] = [""]
    for _ in range(2 * n):
        temp = []
        for x in res:
            if x.count("(") < n:
                temp.append(x + "(")
            if x.count("(") > x.count(")"):
                temp.append(x + ")")
        res = temp

    return res
