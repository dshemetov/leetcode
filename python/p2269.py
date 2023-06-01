# %% 2269. Find The k-Beauty of a Number https://leetcode.com/problems/find-the-k-beauty-of-a-number/
def divisorSubstrings(num: int, k: int) -> int:
    result = 0
    digits = str(num)
    for i in range(len(digits) - k + 1):
        sub = int(digits[i : i + k])
        if sub == 0:
            continue
        if num % sub == 0:
            result += 1
    return result
