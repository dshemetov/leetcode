# %% 6. Zigzag Conversion https://leetcode.com/problems/zigzag-conversion/


# Lessons learned:
# - I went directly for figuring out the indexing patterns, since the matrix
#   approach seemed too boring. After writing out a few examples for numRows =
#   3, 4, 5, I found the pattern.
# - The second solution is a very clever solution from the discussion. It relies
#   on the fact that each new character must be appended to one of the rows, so
#   it just keeps track of which row to append to.
def convert(s: str, numRows: int) -> str:
    """
    Examples:
    >>> convert("PAYPALISHIRING", 1)
    'PAYPALISHIRING'
    >>> convert("PAYPALISHIRING", 2)
    'PYAIHRNAPLSIIG'
    >>> convert("PAYPALISHIRING", 3)
    'PAHNAPLSIIGYIR'
    >>> convert("PAYPALISHIRING", 4)
    'PINALSIGYAHRPI'
    >>> convert("A", 1)
    'A'
    >>> convert("A", 3)
    'A'
    """
    if numRows == 1:
        return s

    if numRows == 2:
        return s[::2] + s[1::2]

    new_s = []
    for i in range(0, numRows):
        if i in {0, numRows - 1}:
            gaps = [2 * numRows - 2, 2 * numRows - 2]
        else:
            gaps = [2 * numRows - 2 - 2 * i, 2 * i]

        ix, j = i, 0

        while ix < len(s):
            new_s += s[ix]
            ix += gaps[j % 2]
            j += 1

    return "".join(new_s)


def convert2(s: str, numRows: int) -> str:
    if numRows == 1:
        return s
    rows = [""] * numRows
    backward = True
    index = 0
    for char in s:
        rows[index] += char
        if index in {0, numRows - 1}:
            backward = not backward
        if backward:
            index -= 1
        else:
            index += 1
    return "".join(rows)
