# %% 45. Jump Game II https://leetcode.com/problems/jump-game-ii/


# Lessons learned:
# - This is the backward version of the dynamic programming solution from
#   problem 55 Jump Game, except here we keep track of the move counts.
# - It turns out that the greedy solution is optimal. The intuition is that we
#   always want to jump to the farthest reachable index. The proof is by
#   contradiction. Suppose we have a better solution that jumps to a closer
#   index. Then we can always replace that jump with a jump to the farthest
#   reachable index, and the new solution will be at least as good as the
#   original one. The only necessary jumps are the ones that allow a new
#   farthest index to be reached.
def jump(nums: list[int]) -> int:
    """
    Examples:
    >>> jump([2,3,1,1,4])
    2
    >>> jump([2,3,0,1,4])
    2
    """
    n = len(nums)
    reachable = [0] * (n - 1) + [1]
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, min(n, i + nums[i] + 1)):
            if reachable[j]:
                reachable[i] = (
                    min(1 + reachable[j], reachable[i])
                    if reachable[i] != 0
                    else 1 + reachable[j]
                )

    return reachable[0] - 1


def jump2(nums: list[int]) -> int:
    # The starting range of the first jump is [0, 0]
    answer, n = 0, len(nums)
    cur_end, cur_far = 0, 0

    for i in range(n - 1):
        # Update the farthest reachable index of this jump.
        cur_far = max(cur_far, i + nums[i])

        # If we finish the starting range of this jump,
        # Move on to the starting range of the next jump.
        if i == cur_end:
            answer += 1
            cur_end = cur_far

    return answer
