# %% 14. Longest Common Prefix https://leetcode.com/problems/longest-common-prefix/
def longest_common_prefix(strs: list[str]) -> str:
    """
    Examples:
    >>> longest_common_prefix(["flower","flow","flight"])
    'fl'
    >>> longest_common_prefix(["dog","racecar","car"])
    ''
    >>> longest_common_prefix(["dog","dog","dog","dog"])
    'dog'
    """
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
        if prefix == "":
            break
    return prefix
