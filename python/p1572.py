# %% 1572. Matrix Diagonal Sum https://leetcode.com/problems/matrix-diagonal-sum/
def diagonalSum(mat: list[list[int]]) -> int:
    """
    Examples:
    >>> diagonalSum([[1,2,3],[4,5,6],[7,8,9]])
    25
    >>> diagonalSum([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    8
    """
    n = len(mat)

    if n == 1:
        return mat[0][0]

    total = 0

    for i in range(n):
        total += mat[i][i] + mat[n - 1 - i][i]

    if n % 2 == 1:
        total -= mat[n // 2][n // 2]

    return total
