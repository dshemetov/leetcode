# %% 433. Minimum Genetic Mutation https://leetcode.com/problems/minimum-genetic-mutation/
def minMutation(startGene: str, endGene: str, bank: list[str]) -> int:
    """
    Examples:
    >>> minMutation("AACCGGTT", "AACCGGTA", ["AACCGGTA"])
    1
    >>> minMutation("AACCGGTT", "AAACGGTA", ["AACCGGTA", "AACCGCTA", "AAACGGTA"])
    2
    >>> minMutation("AAAAACCC", "AACCCCCC", ["AAAACCCC", "AAACCCCC", "AACCCCCC"])
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
