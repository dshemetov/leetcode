# %% 10. Regular Expression Matching https://leetcode.com/problems/regular-expression-matching/


# Lessons learned:
# - I looked at the solution and then wrote it from scratch. The key is the recursive structure
#
#       is_match(s, p) = is_match(s[1:], p[1:])                         if s[0] == p[0] or p[0] == "."
#       is_match(s, p) = is_match(s, p[2:]) or                          if p[1] == "*"
#                        is_match(s[0], p[0]) and is_match(s[1:], p)
#       is_match(s, p) = False                                           otherwise
#
# - With a little work, we can turn this into a dynamic programming solution.
#   Defining dp[i][j] as is_match(s[i:], p[j:]), we have
#
#       dp[i][j] = dp[i+1][j+1]                                         if s[i] == p[j] or p[j] == "."
#       dp[i][j] = dp[i][j+2] or                                        if p[j+1] == "*"
#                  dp[i+1][j] and (s[i+1] == p[j+2] or p[j+2] == ".")
#       dp[i][j] = False                                                otherwise
# - The solution below is bottom-up.
def is_match(s: str, p: str) -> bool:
    """
    Examples:
    >>> is_match("aa", "a")
    False
    >>> is_match("aa", "a*")
    True
    >>> is_match("ab", ".*")
    True
    """
    dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
    dp[len(s)][len(p)] = True

    for i in range(len(s), -1, -1):
        for j in range(len(p) - 1, -1, -1):
            first_match = i < len(s) and (s[i] == p[j] or p[j] == ".")
            if j + 1 < len(p) and p[j + 1] == "*":
                dp[i][j] = dp[i][j + 2] or first_match and dp[i + 1][j]
            else:
                dp[i][j] = first_match and dp[i + 1][j + 1]

    return dp[0][0]
