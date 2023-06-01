# %% 1822. Sign of the Product of an Array https://leetcode.com/problems/sign-of-the-product-of-an-array/
def arraySign(nums: list[int]) -> int:
    """
    Examples:
    >>> arraySign([-1,-2,-3,-4,3,2,1])
    1
    >>> arraySign([1,5,0,2,-3])
    0
    >>> arraySign([-1,1,-1,1,-1])
    -1
    """
    pos = 1
    for n in nums:
        if n < 0:
            pos *= -1
        elif n == 0:
            return 0
    return pos
