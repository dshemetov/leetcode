# %% 901. Online Stock Span https://leetcode.com/problems/online-stock-span/


# Lessons learned:
# - This uses a monotonically decreasing stack (MDS) to keep track of the
#   previous stock prices and their spans.
class StockSpanner:
    """
    Examples:
    >>> obj = StockSpanner()
    >>> obj.next(100)
    1
    >>> obj.next(80)
    1
    >>> obj.next(60)
    1
    >>> obj.next(70)
    2
    >>> obj.next(60)
    1
    >>> obj.next(75)
    4
    >>> obj.next(85)
    6
    """

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        self.stack.append([price, span])
        return span
