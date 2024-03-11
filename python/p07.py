from collections import deque

_empty = object()


class MyCircularQueue:
    def __init__(self, k: int):
        self.lst = [_empty] * k
        self.k = k
        self.front_ix = 0
        self.rear_ix = 0
        self.len = 0

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False

        if len(self) > 0:
            self.rear_ix = (self.rear_ix + 1) % self.k
        self.lst[self.rear_ix] = value
        self.len += 1

        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False

        self.lst[self.front_ix] = _empty
        if len(self) > 1:
            self.front_ix = (self.front_ix + 1) % self.k
        self.len -= 1

        return True

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.lst[self.front_ix]

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.lst[self.rear_ix]

    def isEmpty(self) -> bool:
        return len(self) == 0

    def isFull(self) -> bool:
        return len(self) == self.k

    def __len__(self) -> int:
        return self.len


def p622(cmds, inputs):
    """
    622. Design Circular Queue https://leetcode.com/problems/design-circular-queue/

    Examples:
    >>> cmd = [
    ...     "MyCircularQueue",
    ...     "enQueue",
    ...     "enQueue",
    ...     "enQueue",
    ...     "enQueue",
    ...     "Rear",
    ...     "isFull",
    ...     "deQueue",
    ...     "enQueue",
    ...     "Rear",
    ... ]
    >>> inputs = [[3], [1], [2], [3], [4], [], [], [], [4], []]
    >>> p622(cmd, inputs)
    Running test case:
    None
    True
    True
    True
    False
    3
    True
    True
    True
    4
    >>> cmd = [
    ...     "MyCircularQueue",
    ...     "enQueue",
    ...     "enQueue",
    ...     "deQueue",
    ...     "enQueue",
    ...     "deQueue",
    ...     "enQueue",
    ...     "deQueue",
    ...     "enQueue",
    ...     "deQueue",
    ...     "Front",
    ... ]
    >>> inputs = [[2], [1], [2], [], [3], [], [3], [], [3], [], []]
    >>> p622(cmd, inputs)
    Running test case:
    None
    True
    True
    True
    True
    True
    True
    True
    True
    True
    3
    """
    print("Running test case:")
    for cmd, inp in zip(cmds, inputs):
        if cmd == "MyCircularQueue":
            obj = MyCircularQueue(inp[0])
            print(None)
        elif cmd == "enQueue":
            print(obj.enQueue(inp[0]))
        elif cmd == "deQueue":
            print(obj.deQueue())
        elif cmd == "Front":
            print(obj.Front())
        elif cmd == "Rear":
            print(obj.Rear())
        elif cmd == "isEmpty":
            print(obj.isEmpty())
        elif cmd == "isFull":
            print(obj.isFull())


def p649(senate: str) -> str:
    """
    649. Dota2 Senate https://leetcode.com/problems/dota2-senate/

    Examples:
    >>> p649("RD")
    'Radiant'
    >>> p649("RDD")
    'Dire'
    >>> p649("DDRRR")
    'Dire'
    >>> p649("D")
    'Dire'
    >>> p649("R")
    'Radiant'
    """
    if senate == "":
        return ""

    queue = deque(senate)
    Rcount = queue.count("R")
    Rvetoes, Dvetoes = 0, 0
    while 0 < Rcount < len(queue):
        s = queue.popleft()
        if s == "R":
            if Dvetoes > 0:
                Dvetoes -= 1
                Rcount -= 1
                continue
            Rvetoes += 1
        else:
            if Rvetoes > 0:
                Rvetoes -= 1
                continue
            Dvetoes += 1
        queue.append(s)

    return "Radiant" if queue[0] == "R" else "Dire"


def p658(arr: list[int], k: int, x: int) -> list[int]:
    """
    658. Find k Closest Elements https://leetcode.com/problems/find-k-closest-elements/

    Lessons learned:
    - My solution uses a straightforward binary search to find the closest element
    to x and iterated from there.
    - I include a clever solution from the discussion that uses binary search to
    find the leftmost index of the k closest elements.
    - I had some vague intuition that it could be framed as a minimization
    problem, but I couldn't find the loss function.

    Examples:
    >>> p658([1, 2, 3, 4, 5], 4, 3)
    [1, 2, 3, 4]
    >>> p658([1, 2, 3, 4, 5], 4, -1)
    [1, 2, 3, 4]
    >>> p658([1, 2, 3, 4, 5], 4, 4)
    [2, 3, 4, 5]
    >>> p658([1, 2, 3, 4, 5], 2, 4)
    [3, 4]
    """

    def find_insertion_index(arr: list[int], x: int) -> int:
        lo, hi = 0, len(arr) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == x:
                return mid
            if arr[mid] < x:
                lo = mid + 1
            hi = mid - 1
        return lo

    ix = find_insertion_index(arr, x)
    lst = []
    if ix == 0:
        lst = arr[:k]
    elif ix == len(arr):
        lst = arr[-k:]
    else:
        lo, hi = ix - 1, ix

        while len(lst) < k:
            if lo < 0:
                lst.append(arr[hi])
                hi += 1
            elif hi >= len(arr):
                lst.append(arr[lo])
                lo -= 1
            elif abs(x - arr[lo]) <= abs(x - arr[hi]):
                lst.append(arr[lo])
                lo -= 1
            elif abs(x - arr[lo]) > abs(x - arr[hi]):
                lst.append(arr[hi])
                hi += 1

    return sorted(lst)


def p658_2(arr: list[int], k: int, x: int) -> list[int]:
    """
    Examples:
    >>> p658_2([1, 2, 3, 4, 5], 4, 3)
    [1, 2, 3, 4]
    >>> p658_2([1, 2, 3, 4, 5], 4, -1)
    [1, 2, 3, 4]
    >>> p658_2([1, 2, 3, 4, 5], 4, 4)
    [2, 3, 4, 5]
    >>> p658_2([1, 2, 3, 4, 5], 2, 4)
    [3, 4]
    >>> p658_2([1, 2, 3, 3, 4, 5, 90, 100], 3, 4)
    [3, 3, 4]
    """
    lo, hi = 0, len(arr) - k
    while lo < hi:
        mid = (lo + hi) // 2
        # Equivalently x > (arr[mid] + arr[mid + k]) / 2
        if x - arr[mid] > arr[mid + k] - x:
            lo = mid + 1
        else:
            hi = mid
    return arr[lo : lo + k]
