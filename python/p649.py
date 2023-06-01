# %% 649. Dota2 Senate https://leetcode.com/problems/dota2-senate/
from collections import deque


def predictPartyVictory(senate: str) -> str:
    """
    Examples:
    >>> predictPartyVictory("RD")
    'Radiant'
    >>> predictPartyVictory("RDD")
    'Dire'
    >>> predictPartyVictory("DDRRR")
    'Dire'
    >>> predictPartyVictory("D")
    'Dire'
    >>> predictPartyVictory("R")
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
