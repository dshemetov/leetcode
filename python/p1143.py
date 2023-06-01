# %% 1143. Longest Common Subsequence https://leetcode.com/problems/longest-common-subsequence/


# Lessons learned:
# - This is a classic dynamic programming problem. Define
#
#       dp(i, j) = length of longest common subsequence of text1[:i] and text2[:j]
#
#   The recursion is:
#
#       dp(i, j) = 1 + dp(i - 1, j - 1) if text1[i] == text2[j]
#       dp(i, j) = max(dp(i - 1, j), dp(i, j - 1)) otherwise
#       dp(i, j) = 0 if i == 0 or j == 0
#
# - To avoid recursion, we can use a bottom-up approach, where we start with the
#   smallest subproblems and build up to the largest, storing the results in a
#   table.
def longestCommonSubsequence(text1: str, text2: str) -> int:
    """
    Examples:
    >>> longestCommonSubsequence("abcde", "ace")
    3
    >>> longestCommonSubsequence("abc", "abc")
    3
    >>> longestCommonSubsequence("abc", "def")
    0
    """
    dp_ = [[0 for _ in range(len(text2) + 1)] for _ in range(len(text1) + 1)]

    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            if text1[i - 1] == text2[j - 1]:
                dp_[i][j] = 1 + dp_[i - 1][j - 1]
            else:
                dp_[i][j] = max(dp_[i - 1][j], dp_[i][j - 1])

    return dp_[-1][-1]
