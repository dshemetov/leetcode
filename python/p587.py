# %% 587. Erect the Fence https://leetcode.com/problems/erect-the-fence/
import math
from collections import defaultdict
from fractions import Fraction
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


# Lessons learned:
# - A broad class of computational geometry algorithms solve this:
#   https://en.wikipedia.org/wiki/Convex_hull_algorithms#Algorithms
# - The Graham scan is easy to understand and decently fast:
#   https://en.wikipedia.org/wiki/Graham_scan
# - Tip from a graphics guy: avoid representing angles with degrees/radians,
#   stay in fractions. This avoids numerical issues with floating points, but
#   it's not without its own problems.
# - The atan2 function was invented back in the Fortran days and makes for a
#   stable polar angle definition. It's also fast.
# - The edge-cases of the Graham scan are tricky, especially all the cases with
#   colinear points.
def ccw(p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int]) -> float:
    """
    Examples:
    >>> ccw((0, 0), (1, 0), (0, 1))
    1.0
    >>> ccw((0, 0), (1, 0), (1, 1))
    1.0
    >>> ccw((0, 0), (1, 0), (1, 0))
    0.0
    >>> ccw((0, 0), (1, 0), (0, -1))
    -1.0
    """
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def polar_angle(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Get the polar angle of the vector from p1 to p2."""
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    return np.arctan2(v1[1], v1[0])


def point_sorter(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Sort by polar angle and break ties by distance."""
    return (polar_angle(p1, p2), -np.linalg.norm((p2[0] - p1[0], p2[1] - p1[1])))


def atan2notan(y: int, x: int) -> Fraction:
    """A polar angle substitute without trigonometry or floating points.

    Imagine tracing out a circle counterclockwise and measuring the angle to the tracing vector
    from the positive x axis. This is the sorted order we wish to achieve. This function will give
    a lexically smaller tuple for smaller angles.
    """
    if x == 0 and y == 0:
        return (0, 0)
    if x > 0 and y >= 0:
        return (0, Fraction(y, x))
    if x == 0 and y > 0:
        return (1, 0)
    if x < 0:
        return (2, Fraction(y, x))
    if x == 0 and y < 0:
        return (3, 0)
    if y < 0 < x:
        return (4, Fraction(y, x))
    raise ValueError("How did you even get here?")


def partition_by(l: list, f: Callable) -> dict:
    """Partition a list into lists based on a predicate."""
    d = defaultdict(list)
    for item in l:
        d[f(item)].append(item)
    return d


def plot_points(points: list[tuple[int, int]], hull: list[tuple[int, int]]):
    x, y = zip(*points)
    plt.scatter(x, y)
    x, y = zip(*hull)
    plt.plot(x, y, color="green")
    plt.show()


def outerTrees(trees: list[list[int]]) -> list[list[int]]:
    """
    We are going to use a Graham scan to find the convex hull of the points.

    Examples:
    >>> outerTrees([[1,1],[2,2],[2,0],[2,4],[3,3],[4,2]])
    [[2, 0], [4, 2], [3, 3], [2, 4], [1, 1]]
    >>> outerTrees([[1,2],[2,2],[4,2]])
    [[1, 2], [2, 2], [4, 2]]
    >>> outerTrees([[1,1],[2,2],[3,3],[2,1],[4,1],[2,3],[1,3]])
    [[1, 1], [2, 1], [4, 1], [3, 3], [2, 3], [1, 3]]
    >>> outerTrees([[3,0],[4,0],[5,0],[6,1],[7,2],[7,3],[7,4],[6,5],[5,5],[4,5],[3,5],[2,5],[1,4],[1,3],[1,2],[2,1],[4,2],[0,3]])
    [[3, 0], [4, 0], [5, 0], [6, 1], [7, 2], [7, 3], [7, 4], [6, 5], [5, 5], [4, 5], [3, 5], [2, 5], [1, 4], [0, 3], [1, 2], [2, 1]]
    >>> outerTrees([[0,2],[0,1],[0,0],[1,0],[2,0],[1,1]])
    [[0, 0], [1, 0], [2, 0], [1, 1], [0, 2], [0, 1]]
    >>> outerTrees([[0,2],[0,4],[0,5],[0,9],[2,1],[2,2],[2,3],[2,5],[3,1],[3,2],[3,6],[3,9],[4,2],[4,5],[5,8],[5,9],[6,3],[7,9],[8,1],[8,2],[8,5],[8,7],[9,0],[9,1],[9,6]])
    [[9, 0], [9, 1], [9, 6], [7, 9], [5, 9], [3, 9], [0, 9], [0, 5], [0, 4], [0, 2], [2, 1]]
    >>> outerTrees([[0,0],[0,1],[0,2],[1,2],[2,2],[3,2],[3,1],[3,0],[2,0],[1,0],[1,1],[3,3]])
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [3, 3], [0, 2], [0, 1]]
    """
    lowest_left_point = (math.inf, math.inf)
    for x, y in trees:
        if y < lowest_left_point[1] or (
            y == lowest_left_point[1] and x < lowest_left_point[0]
        ):
            lowest_left_point = (x, y)

    trees_by_slope = partition_by(
        trees,
        lambda p: atan2notan(p[1] - lowest_left_point[1], p[0] - lowest_left_point[0]),
    )
    slopes = sorted(trees_by_slope.keys())

    # Handles many colinear cases; order doesn't matter for leetcode
    if len(slopes) == 1:
        return trees

    def distance(p1, p2):
        return np.linalg.norm((p1[1] - p2[1], p1[0] - p2[0]))

    # The right-most line should sort by increasing distance from lowest left point
    trees_by_slope[slopes[0]] = sorted(
        trees_by_slope[slopes[0]], key=lambda p: distance(p, lowest_left_point)
    )
    # The left-most line should sort by decreasing distance from lowest left point
    trees_by_slope[slopes[-1]] = sorted(
        trees_by_slope[slopes[-1]], key=lambda p: -distance(p, lowest_left_point)
    )
    # The rest should sort by increasing distance from lowest left point
    for slope in slopes[1:-1]:
        trees_by_slope[slope] = sorted(
            trees_by_slope[slope], key=lambda p: distance(p, lowest_left_point)
        )

    stack = []
    for slope in slopes:
        for tree in trees_by_slope[slope]:
            while len(stack) >= 2 and ccw(stack[-2], stack[-1], tree) < 0:
                stack.pop()
            stack.append(tree)

    return stack
