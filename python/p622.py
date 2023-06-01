# %% 622. Design Circular Queue https://leetcode.com/problems/design-circular-queue/
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


def run(cmds, inputs):
    """
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
    >>> run(cmd, inputs)
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
    >>> run(cmd, inputs)
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
