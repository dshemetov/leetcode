def p402(num: str, k: int) -> str:
    """
    402. Remove k Digits https://leetcode.com/problems/remove-k-digits/

    Lessons learned:
    - try to build up a heuristic algorithm from a few examples

    Examples:
    >>> p402("1432219", 3)
    '1219'
    >>> p402("10200", 1)
    '200'
    >>> p402("10", 2)
    '0'
    >>> p402("9", 1)
    '0'
    >>> p402("112", 1)
    '11'
    """
    if len(num) <= k:
        return "0"

    stack = []
    for c in num:
        if c == "0" and not stack:
            continue
        while stack and stack[-1] > c and k > 0:
            stack.pop()
            k -= 1
        stack.append(c)

    if k > 0:
        stack = stack[:-k]

    return "".join(stack).lstrip("0") or "0"


def p433(startGene: str, endGene: str, bank: list[str]) -> int:
    """
    433. Minimum Genetic Mutation https://leetcode.com/problems/minimum-genetic-mutation/

    Examples:
    >>> p433("AACCGGTT", "AACCGGTA", ["AACCGGTA"])
    1
    >>> p433("AACCGGTT", "AAACGGTA", ["AACCGGTA", "AACCGCTA", "AAACGGTA"])
    2
    >>> p433("AAAAACCC", "AACCCCCC", ["AAAACCCC", "AAACCCCC", "AACCCCCC"])
    3
    """

    def get_mutations(gene: str, bank: set[str]) -> set[str]:
        return {
            mutation
            for mutation in bank
            if sum(1 for i in range(len(mutation)) if mutation[i] != gene[i]) == 1
        }

    bank = set(bank)
    explored = set()
    unexplored = set({startGene})
    steps = 0
    while unexplored:
        new_unexplored = set()
        for gene in unexplored:
            if gene == endGene:
                return steps
            explored |= {gene}
            for mutations in get_mutations(gene, bank):
                if mutations not in explored:
                    new_unexplored |= {mutations}
        unexplored = new_unexplored
        bank -= explored
        steps += 1

    return -1


def p456(nums: list[int]) -> bool:
    """
    456. 132 Pattern https://leetcode.com/problems/132-pattern/

    Lessons learned:
    - Another opportunity to put monotonic stacks to use. I still don't know
    exactly when to use them, probably need some more practice.

    Examples:
    >>> p456([1, 2, 3, 4])
    False
    >>> p456([3, 1, 4, 2])
    True
    >>> p456([-1, 3, 2, 0])
    True
    >>> p456([1, 2, 0, 3, -1, 4, 2])
    True
    >>> p456([1, 3, -1, 1, 1])
    False
    >>> p456([-2, 1, -2])
    False
    """
    span_stack = [[nums[0], nums[0]]]

    for new_val in nums[1:]:
        top_span = span_stack.pop()
        if new_val < top_span[0]:
            span_stack.append(top_span)
            span_stack.append([new_val, new_val])
        elif top_span[0] < new_val < top_span[1]:
            return True
        elif new_val > top_span[1]:
            top_span[1] = new_val
            while span_stack:
                next_span = span_stack.pop()
                if top_span[1] <= next_span[0]:
                    span_stack.append(next_span)
                    break
                if next_span[0] < top_span[1] < next_span[1]:
                    return True
                if next_span[1] <= top_span[1]:
                    continue
            span_stack.append(top_span)
        else:
            span_stack.append(top_span)
    return False


def p495(timeSeries: list[int], duration: int) -> int:
    """
    495. Teemo Attacking https://leetcode.com/problems/teemo-attacking

    Examples:
    >>> p495([1,4], 2)
    4
    >>> p495([1,2], 2)
    3
    """
    total_duration = 0
    for i in range(1, len(timeSeries)):
        time_delta = timeSeries[i] - timeSeries[i - 1]
        total_duration += min(duration, time_delta)
    return total_duration + duration
