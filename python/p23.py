def p2215(nums1: list[int], nums2: list[int]) -> list[list[int]]:
    """
    2215. Find the Difference of Two Arrays https://leetcode.com/problems/find-the-difference-of-two-arrays/

    Examples:
    >>> p2215([1,2,3], [2,4,6])
    [[1, 3], [4, 6]]
    >>> p2215([1,2,3,3], [1,1,2,2])
    [[3], []]
    """
    s1, s2 = set(nums1), set(nums2)
    return [[n for n in s1 if n not in s2], [n for n in s2 if n not in s1]]


def p2269(num: int, k: int) -> int:
    """
    2269. Find The k-Beauty of a Number https://leetcode.com/problems/find-the-k-beauty-of-a-number/
    """
    result = 0
    digits = str(num)
    for i in range(len(digits) - k + 1):
        sub = int(digits[i : i + k])
        if sub == 0:
            continue
        if num % sub == 0:
            result += 1
    return result
