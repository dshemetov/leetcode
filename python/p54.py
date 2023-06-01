# %% 54. Spiral Matrix https://leetcode.com/problems/spiral-matrix/
from itertools import cycle


def spiralOrder(matrix: list[list[int]]) -> list[int]:
    """
    Examples:
    >>> spiralOrder([[1,2,3],[4,5,6],[7,8,9]])
    [1, 2, 3, 6, 9, 8, 7, 4, 5]
    >>> spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
    >>> spiralOrder([[1]])
    [1]
    >>> spiralOrder([[1,2],[3,4]])
    [1, 2, 4, 3]
    """
    m, n = len(matrix), len(matrix[0])
    directions = cycle([[0, 1], [1, 0], [0, -1], [-1, 0]])
    cur_pos, steps = (0, 0), 1
    visited, elements = set(), [matrix[0][0]]
    visited.add(cur_pos)
    for dm, dn in directions:
        if steps == m * n:
            break

        while True:
            new_pos = (cur_pos[0] + dm, cur_pos[1] + dn)

            if new_pos in visited or not (0 <= new_pos[0] < m and 0 <= new_pos[1] < n):
                break

            cur_pos = new_pos
            elements.append(matrix[cur_pos[0]][cur_pos[1]])
            visited.add(cur_pos)
            steps += 1

    return elements
