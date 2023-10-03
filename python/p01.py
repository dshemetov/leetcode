import math
from collections import defaultdict, deque
from itertools import cycle, product

import numpy as np


class ListNode:
    def __init__(self, val=0, next=None):  # pylint: disable=redefined-builtin
        self.val = val
        self.next = next


def list_to_listnode(lst: list[int]) -> ListNode | None:
    if not lst:
        return None

    original_head = head = ListNode(lst[0])
    for x in lst[1:]:
        head.next = ListNode(x)
        head = head.next

    return original_head


def listnode_to_list(head: ListNode) -> list[int]:
    """
    Examples:
    >>> listnode_to_list(list_to_listnode([1, 2, 3, 4, 5]))
    [1, 2, 3, 4, 5]
    """
    lst = []
    while head:
        lst.append(head.val)
        head = head.next

    return lst


def int_to_list(i: int) -> ListNode:
    num_list = None
    for x in str(i):
        num_list = ListNode(val=int(x), next=num_list)
    return num_list


def list_to_int(lst: ListNode) -> int:
    """
    Examples:
    >>> list_to_int(int_to_list(0))
    0
    >>> list_to_int(int_to_list(2))
    2
    >>> list_to_int(int_to_list(12))
    12
    >>> list_to_int(int_to_list(15))
    15
    >>> list_to_int(int_to_list(255))
    255
    """
    num = 0
    digit = 0
    while lst:
        num += lst.val * 10**digit
        digit += 1
        lst = lst.next
    return num


def p2(l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
    """
    2. Add Two Numbers https://leetcode.com/problems/add-two-numbers/

    Examples:
    >>> list_to_int(p2(int_to_list(0), int_to_list(15)))
    15
    >>> list_to_int(p2(int_to_list(12), int_to_list(15)))
    27
    >>> list_to_int(p2(int_to_list(12), int_to_list(153)))
    165
    """
    first_node = ListNode(0)
    cur_node = first_node
    carry_bit = 0
    while l1 or l2 or carry_bit > 0:
        x = l1.val if l1 else 0
        y = l2.val if l2 else 0

        num = x + y + carry_bit
        carry_bit = num // 10

        new_node = ListNode(num % 10)
        cur_node.next = new_node
        cur_node = new_node

        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None

    return first_node.next


def p3(s: str) -> int:
    """
    3. Longest Substring Without Repeating Characters https://leetcode.com/problems/longest-substring-without-repeating-characters/

    Examples:
    >>> p3("a")
    1
    >>> p3("aa")
    1
    >>> p3("aaa")
    1
    >>> p3("aab")
    2
    >>> p3("abba")
    2
    >>> p3("abccba")
    3
    >>> p3("au")
    2
    >>> p3("cdd")
    2
    >>> p3("abcabcbb")
    3
    >>> p3("aabcdef")
    6
    >>> p3("abcdeffff")
    6
    >>> p3("dvdf")
    3
    >>> p3("ohomm")
    3
    """
    if not s:
        return 0

    longest = 1
    lo = 0
    hi = 1
    char_set = set(s[lo])
    while hi < len(s):
        if s[hi] not in char_set:
            char_set.add(s[hi])
            hi += 1
            longest = max(longest, hi - lo)
        else:
            char_set.remove(s[lo])
            lo += 1

    return longest


def p4(nums1: list[int], nums2: list[int]) -> float:
    """
    4. Median of Two Sorted Arrays https://leetcode.com/problems/median-of-two-sorted-arrays/

    Lessons learned:
    - I spent weeks thinking about this problem before giving up and looking for
    a solution.
    - There are a few key insights to this problem. First, the median has the
    property of being the partition point where half the elements are less and
    half are greater. Second, a partition point in one array implies a partition
    point in the other array, which means we can find the partition point via
    binary search on one array.
    - We use the following notation in the code:

        A refers to the shorter array,
        B refers to the longer array,
        midA refers to a partition point in A,
        midB refers to a partition point in B,
        Aleft = A[midA - 1], refers to the largest element in the left partition of A
        Aright = A[midA], refers to the smallest element in the right partition of A
        Bleft = B[midB - 1], refers to the largest element in the left partition of B
        Bright = B[midB], refers to the smallest element in the right partition of B

    - To expand more on the second insight, consider the following example:

        A = [1, 3, 5, 7, 9], B = [2, 4, 6, 8, 10, 12, 14, 16]

    Suppose we choose midA = 4. Since the total number of elements is 13, half
    of which is 6.5, then, breaking the tie arbitrarily, 7 elements must be in
    the left partition and 6 elements must be in the right partition. Since 4
    elements are already in the left partition, we need to add 3 more elements
    to the left partition, which we can do choosing midB = 3. This corresponds
    to the total left partition [1, 2, 3, 4, 5, 6, 7] and the total right
    partition [8, 9, 10, 12, 14, 16].
    - In general, we have

        midA + midB = (len(A) + len(B) + 1) // 2,

    which implies

        midB = (len(A) + len(B) + 1) // 2 - midA.

    - Note that having the +1 inside the divfloor covers the cases correctly for
    odd and even total number of elements. For example, if the total number of
    elements is 13 and i = 4, then j = (13 + 1) // 2 - 4 = 3, which is correct.
    If the total number of elements is 12 and i = 4, then j = (12 + 1) // 2 - 4
    = 2, which is also correct. If the +1 was not inside the divfloor, then the
    second case would be incorrect.
    - So our problem is solved if we can find a partition (midA, midB) with:

        len(A[:midA]) + len(B[:midB]) == len(A[midA:]) + len(B[midB:]),
        Aleft <= Bright,
        Bleft <= Aright.

    - The median is then

        median = max(Aleft, Bleft)                               if len(A) + len(B) odd
                = (max(Aleft, Bleft) + min(Aright, Bright)) / 2.  else

    - Swapping two variables in Python swaps pointers under the hood:
    https://stackoverflow.com/a/62038590/4784655.

    Examples:
    >>> p4([1, 3], [2])
    2.0
    >>> p4([1, 2], [3, 4])
    2.5
    >>> p4([1, 3], [2, 4])
    2.5
    >>> a1 = [5, 13, 15]
    >>> b1 = [0, 10, 10, 15, 20, 20, 25]
    >>> p4(a1, b1) == get_median_sorted(sorted(a1 + b1))
    True
    >>> a2 = [9, 36, 44, 45, 51, 67, 68, 69]
    >>> b2 = [7, 20, 26, 27, 30, 43, 54, 73, 76, 88, 91, 94]
    >>> p4(a2, b2) == get_median_sorted(sorted(a2 + b2))
    True
    >>> a2 = [2, 2, 2, 2, 2, 2, 5]
    >>> b2 = [0, 0, 0, 0, 0, 0, 0, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    >>> p4(a2, b2) == get_median_sorted(sorted(a2 + b2))
    True
    >>> a2 = [2, 2, 2, 4, 5, 7, 8, 9]
    >>> b2 = [1, 1, 1, 1, 1, 3, 6, 10, 11, 11, 11, 11]
    >>> p4(a2, b2) == get_median_sorted(sorted(a2 + b2))
    True
    """
    if not nums1:
        return get_median_sorted(nums2)
    if not nums2:
        return get_median_sorted(nums1)

    A, B = nums1, nums2

    if len(A) > len(B):
        A, B = B, A

    total = len(A) + len(B)
    lo, hi = 0, len(A)
    while True:
        midA = (lo + hi) // 2
        midB = (total + 1) // 2 - midA

        Aleft = A[midA - 1] if midA > 0 else float("-inf")
        Aright = A[midA] if midA < len(A) else float("inf")
        Bleft = B[midB - 1] if midB > 0 else float("-inf")
        Bright = B[midB] if midB < len(B) else float("inf")

        if Aleft <= Bright and Bleft <= Aright:
            if total % 2 == 0:
                return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
            return float(max(Aleft, Bleft))

        if Aleft > Bright:
            hi = midA - 1
        else:
            lo = midA + 1


def get_median_sorted(nums: list[int]) -> float:
    if len(nums) == 1:
        return nums[0]

    mid = len(nums) // 2

    if len(nums) % 2 == 0:
        return (nums[mid] + nums[mid - 1]) / 2

    return nums[mid]


def p5(s: str) -> str:
    """
    5. Longest Palindromic Substring https://leetcode.com/problems/longest-palindromic-substring/

    Lessons learned:
    - I tried an approach with three pointers and expanding outwards if the
    characters matched. The edge case that stumped me was handling long runs of
    the same character such as "aaaaaaaaa". The issue there is that you need to
    keep changing the palindrome center. I gave up on that approach and looked
    at the solution.
    - The solution is straightforward and I probably would have thought of it,
    if I didn't get stuck trying to fix the three pointer approach.

    Examples:
    >>> p5("babad")
    'bab'
    >>> p5("cbbd")
    'bb'
    >>> p5("ac")
    'a'
    >>> p5("abcde")
    'a'
    >>> p5("abcdeedcba")
    'abcdeedcba'
    >>> p5("abcdeeffdcba")
    'ee'
    >>> p5("abaaba")
    'abaaba'
    >>> p5("abaabac")
    'abaaba'
    >>> p5("aaaaa")
    'aaaaa'
    >>> p5("aaaa")
    'aaaa'
    """
    if len(s) == 1:
        return s

    lo, hi = 0, 1
    max_length = 1
    res_string = s[0]

    def expand_center(lo, hi):
        while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
            lo -= 1
            hi += 1
        return lo + 1, hi - 1

    for i in range(1, len(s)):
        lo, hi = expand_center(i - 1, i + 1)
        if hi - lo + 1 > max_length:
            max_length = hi - lo + 1
            res_string = s[lo : hi + 1]

        lo, hi = expand_center(i - 1, i)
        if hi - lo + 1 > max_length:
            max_length = hi - lo + 1
            res_string = s[lo : hi + 1]

    return res_string


def p6(s: str, numRows: int) -> str:
    """
    6. Zigzag Conversion https://leetcode.com/problems/zigzag-conversion/

    Lessons learned:
    - I went directly for figuring out the indexing patterns, since the matrix
    approach seemed too boring. After writing out a few examples for numRows =
    3, 4, 5, I found the pattern.
    - The second solution is a very clever solution from the discussion. It relies
    on the fact that each new character must be appended to one of the rows, so
    it just keeps track of which row to append to.

    Examples:
    >>> p6("PAYPALISHIRING", 1)
    'PAYPALISHIRING'
    >>> p6("PAYPALISHIRING", 2)
    'PYAIHRNAPLSIIG'
    >>> p6("PAYPALISHIRING", 3)
    'PAHNAPLSIIGYIR'
    >>> p6("PAYPALISHIRING", 4)
    'PINALSIGYAHRPI'
    >>> p6("A", 1)
    'A'
    >>> p6("A", 3)
    'A'
    """
    if numRows == 1:
        return s

    if numRows == 2:
        return s[::2] + s[1::2]

    new_s = []
    for i in range(0, numRows):
        if i in {0, numRows - 1}:
            gaps = [2 * numRows - 2, 2 * numRows - 2]
        else:
            gaps = [2 * numRows - 2 - 2 * i, 2 * i]

        ix, j = i, 0

        while ix < len(s):
            new_s += s[ix]
            ix += gaps[j % 2]
            j += 1

    return "".join(new_s)


def p6_2(s: str, numRows: int) -> str:
    if numRows == 1:
        return s
    rows = [""] * numRows
    backward = True
    index = 0
    for char in s:
        rows[index] += char
        if index in {0, numRows - 1}:
            backward = not backward
        if backward:
            index -= 1
        else:
            index += 1
    return "".join(rows)


def p7(x: int) -> int:
    """
    7. Reverse Integer https://leetcode.com/problems/reverse-integer/

    Lessons learned:
    - The most interesting part of this problem is finding out how to check for
    overflow without overflowing. This can be done by checking whether the
    multiplication by 10 will overflow or if the the multiplication by 10 will
    bring you right to the edge and the next digit will overflow.
    - Another interesting part is that Python's modulo operator behaves
    differently than in C. The modulo operator performs Euclidean division a = b
    * q + r, where r is the remainder and q is the quotient. In Python, the
    remainder r is always positive, whereas in C, the remainder r has the same
    sign as the dividend a. This in turn, implies that in C, q = truncate(a/b),
    while in Python, q = floor(a/b). So in Python, -(-x % n) = -((n - x % n) %
    n), while in C, we have (-x % n) = -(x % n). Also, in Python, -(-x // n) =
    (x - 1) // n + 1.

    Examples:
    >>> p7(123)
    321
    >>> p7(-123)
    -321
    >>> p7(120)
    21
    >>> p7(-1563847412)
    0
    >>> p7(-10)
    -1
    """
    int_max_div10 = (2**31 - 1) // 10
    int_min_div10 = (-(2**31)) // 10 + 1

    rx = 0
    while x != 0:
        if x < 0:
            r = -((10 - x % 10) % 10)
            x = (x - 1) // 10 + 1
        else:
            r = x % 10
            x //= 10

        if rx > int_max_div10 or (rx == int_max_div10 and r > 7):
            return 0
        if rx < int_min_div10 or (rx == int_min_div10 and r < -8):
            return 0

        rx = rx * 10 + r

    return rx


def p8(s: str) -> int:
    """
    8. String to Integer (atoi) https://leetcode.com/problems/string-to-integer-atoi/

    Examples:
    >>> p8("42")
    42
    >>> p8("   -42")
    -42
    >>> p8("4193 with words")
    4193
    >>> p8("words and 987")
    0
    >>> p8("-91283472332")
    -2147483648
    >>> p8("91283472332")
    2147483647
    >>> p8("3.14159")
    3
    >>> p8("+-2")
    0
    >>> p8("  -0012a42")
    -12
    >>> p8("  +0 123")
    0
    >>> p8("-0")
    0
    """
    _max = 2**31 - 1
    _min = 2**31

    factor = 0
    num = 0
    digits_started = False
    for c in s:
        if not digits_started:
            if c == " ":
                continue
            if c == "-":
                factor = -1
                digits_started = True
            elif c == "+":
                factor = 1
                digits_started = True
            elif c.isdigit():
                factor = 1
                digits_started = True
                num = int(c)
            else:
                return 0
        else:
            if c.isdigit():
                num = num * 10 + int(c)
                if factor == 1 and num > _max:
                    return _max
                if factor == -1 and num > _min:
                    return -_min
            else:
                break

    return factor * num


def p9(x: int) -> bool:
    """
    9. Palindrome Number https://leetcode.com/problems/palindrome-number/

    Examples:
    >>> p9(121)
    True
    >>> p9(-121)
    False
    >>> p9(10)
    False
    >>> p9(0)
    True
    >>> p9(1)
    True
    >>> p9(1331)
    True
    >>> p9(1332)
    False
    >>> p9(133454331)
    True
    >>> p9(1122)
    False
    """
    if x < 0:
        return False
    if x == 0:
        return True

    num_digits = math.floor(math.log10(x)) + 1

    if num_digits == 1:
        return True

    for i in range(num_digits // 2):
        if x // 10**i % 10 != x // 10 ** (num_digits - i - 1) % 10:
            return False
    return True


def p10(s: str, p: str) -> bool:
    """
    10. Regular Expression Matching https://leetcode.com/problems/regular-expression-matching/

    Lessons learned:
    - I looked at the solution and then wrote it from scratch. The key is the recursive structure

        is_match(s, p) = is_match(s[1:], p[1:])                         if s[0] == p[0] or p[0] == "."
        is_match(s, p) = is_match(s, p[2:]) or                          if p[1] == "*"
                        is_match(s[0], p[0]) and is_match(s[1:], p)
        is_match(s, p) = False                                           otherwise

    - With a little work, we can turn this into a dynamic programming solution.
    Defining dp[i][j] as is_match(s[i:], p[j:]), we have

        dp[i][j] = dp[i+1][j+1]                                         if s[i] == p[j] or p[j] == "."
        dp[i][j] = dp[i][j+2] or                                        if p[j+1] == "*"
                    dp[i+1][j] and (s[i+1] == p[j+2] or p[j+2] == ".")
        dp[i][j] = False                                                otherwise
    - The solution below is bottom-up.

    Examples:
    >>> p10("aa", "a")
    False
    >>> p10("aa", "a*")
    True
    >>> p10("ab", ".*")
    True
    """
    dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
    dp[len(s)][len(p)] = True

    for i in range(len(s), -1, -1):
        for j in range(len(p) - 1, -1, -1):
            first_match = i < len(s) and (s[i] == p[j] or p[j] == ".")
            if j + 1 < len(p) and p[j + 1] == "*":
                dp[i][j] = dp[i][j + 2] or first_match and dp[i + 1][j]
            else:
                dp[i][j] = first_match and dp[i + 1][j + 1]

    return dp[0][0]


def p11(height: list[int]) -> int:
    """
    11. Container With Most Water https://leetcode.com/problems/container-with-most-water/

    Lessons learned:
    - The trick to the O(n) solution relies on the following insight: if we
    shorten the container but change the height of the larger side, the area
    will not increase. Therefore, we can start with the widest possible
    container and do at most one comparison per index.
    - This feels like a trick problem and I didn't feel like I learned much from
    it.

    Examples:
    >>> p11([1,8,6,2,5,4,8,3,7])
    49
    >>> p11([1,1])
    1
    """
    lo, hi = 0, len(height) - 1
    m = float("-inf")
    while lo < hi:
        m = max(m, min(height[lo], height[hi]) * (hi - lo))
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1
    return m


def p12(num: int) -> str:
    """
    12. Integer to Roman https://leetcode.com/problems/integer-to-roman/

    Examples:
    >>> p12(3)
    'III'
    >>> p12(4)
    'IV'
    >>> p12(9)
    'IX'
    >>> p12(58)
    'LVIII'
    >>> p12(1994)
    'MCMXCIV'
    """
    letter_map = {
        1: "I",
        4: "IV",
        5: "V",
        9: "IX",
        10: "X",
        40: "XL",
        50: "L",
        90: "XC",
        100: "C",
        400: "CD",
        500: "D",
        900: "CM",
        1000: "M",
    }
    s = ""
    for k in sorted(letter_map.keys(), reverse=True):
        r, num = divmod(num, k)
        s += letter_map[k] * r

    return s


def p13(s: str) -> int:
    """
    13. Roman to Integer https://leetcode.com/problems/roman-to-integer/

    Examples:
    >>> p13("III")
    3
    >>> p13("IV")
    4
    >>> p13("IX")
    9
    >>> p13("LVIII")
    58
    >>> p13("MCMXCIV")
    1994
    """
    d = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    total = 0
    for i in range(len(s) - 1):
        if d[s[i]] < d[s[i + 1]]:
            total -= d[s[i]]
        else:
            total += d[s[i]]

    return total + d[s[i + 1]]


def p14(strs: list[str]) -> str:
    """
    14. Longest Common Prefix https://leetcode.com/problems/longest-common-prefix/

    Examples:
    >>> p14(["flower","flow","flight"])
    'fl'
    >>> p14(["dog","racecar","car"])
    ''
    >>> p14(["dog","dog","dog","dog"])
    'dog'
    """
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
        if prefix == "":
            break
    return prefix


def p15(nums: list[int]) -> list[list[int]]:
    """
    15. 3Sum https://leetcode.com/problems/3sum/

    Examples:
    >>> p15([-1,0,1,2,-1,-4])
    [[-1, -1, 2], [-1, 0, 1]]
    >>> p15([0, 1, 1])
    []
    >>> p15([0, 0, 0])
    [[0, 0, 0]]
    """
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:
                lo += 1
            elif s > 0:
                hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])

                while lo < hi and nums[lo] == nums[lo + 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi - 1]:
                    hi -= 1

                lo += 1
                hi -= 1

    return res


def p16(nums: list[int], target: int) -> int:
    """
    16. 3Sum Closest https://leetcode.com/problems/3sum-closest/

    Examples:
    >>> p16([-1,2,1,-4], 1)
    2
    >>> p16([0,0,0], 1)
    0
    """
    nums.sort()
    res = float("inf")
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi] - target
            res = s if abs(s) < abs(res) else res
            if s < 0:
                lo += 1
            elif s > 0:
                hi -= 1
            else:
                return target

    return res + target


def p17(digits: str) -> list[str]:
    """
    17. Letter Combinations of a Phone Number https://leetcode.com/problems/letter-combinations-of-a-phone-number/

    Lessons learned:
    - Implement the Cartesian product, they say! It's fun, they say! And it turns
    out, that it is kinda fun.
    - The second solution is a little more manual, but worth knowing.


    Examples:
    >>> p17("23")
    ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
    >>> p17("")
    []
    >>> p17("2")
    ['a', 'b', 'c']
    """
    if not digits:
        return []

    letter_map = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }

    res = product(*[letter_map[c] for c in digits])
    return ["".join(t) for t in res]


def p17_2(digits: str) -> list[str]:
    """
    Examples:
    >>> p17_2("23")
    ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
    >>> p17_2("")
    []
    >>> p17_2("2")
    ['a', 'b', 'c']
    """
    if not digits:
        return []

    letter_map = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }

    res = []
    for c in digits:
        if not res:
            res = list(letter_map[c])
        else:
            res = [a + b for a in res for b in letter_map[c]]

    return res


def p18(nums: list[int], target: int) -> list[list[int]]:
    """
    18. 4Sum https://leetcode.com/problems/4sum/

    Lessons learned:
    - The idea is the same as in 3Sum, but with an extra index.

    Examples:
    >>> p18([1,0,-1,0,-2,2], 0)
    [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
    >>> p18([2,2,2,2,2], 8)
    [[2, 2, 2, 2]]
    >>> p18([-2,-1,-1,1,1,2,2], 0)
    [[-2, -1, 1, 2], [-1, -1, 1, 1]]
    """
    nums.sort()
    res = []
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            lo, hi = j + 1, len(nums) - 1
            while lo < hi:
                s = nums[i] + nums[j] + nums[lo] + nums[hi]
                if s < target:
                    lo += 1
                elif s > target:
                    hi -= 1
                else:
                    res.append([nums[i], nums[j], nums[lo], nums[hi]])

                    while lo < hi and nums[lo] == nums[lo + 1]:
                        lo += 1
                    while lo < hi and nums[hi] == nums[hi - 1]:
                        hi -= 1

                    lo += 1
                    hi -= 1

    return res


def p19(head: ListNode | None, n: int) -> ListNode | None:
    """
    19. Remove Nth Node From End of List https://leetcode.com/problems/remove-nth-node-from-end-of-list/

    Examples:
    >>> listnode_to_list(p19(list_to_listnode([1, 2, 3, 4, 5]), 1))
    [1, 2, 3, 4]
    >>> listnode_to_list(p19(list_to_listnode([1, 2, 3, 4, 5]), 2))
    [1, 2, 3, 5]
    >>> listnode_to_list(p19(list_to_listnode([1, 2, 3, 4, 5]), 3))
    [1, 2, 4, 5]
    >>> listnode_to_list(p19(list_to_listnode([1, 2, 3, 4, 5]), 4))
    [1, 3, 4, 5]
    >>> listnode_to_list(p19(list_to_listnode([1, 2, 3, 4, 5]), 5))
    [2, 3, 4, 5]
    >>> listnode_to_list(p19(list_to_listnode([1]), 1))
    []
    >>> listnode_to_list(p19(list_to_listnode([1, 2]), 1))
    [1]
    >>> listnode_to_list(p19(list_to_listnode([1, 2]), 2))
    [2]
    """
    sz = 0
    node = head
    while node:
        node = node.next
        sz += 1

    if sz == 1:
        return None

    if sz == n:
        return head.next

    node = head
    for _ in range(sz - n - 1):
        node = node.next

    node.next = node.next.next

    return head


def p20(s: str) -> bool:
    """
    20. Valid Parentheses https://leetcode.com/problems/valid-parentheses/

    Examples:
    >>> p20("()")
    True
    >>> p20("()[]{}")
    True
    >>> p20("(]")
    False
    """
    stack = deque()
    bracket_map = {"(": ")", "[": "]", "{": "}"}
    for c in s:
        if c in bracket_map:
            stack.append(c)
        elif not stack or bracket_map[stack.pop()] != c:
            return False

    return not stack


def p21(list1: ListNode | None, list2: ListNode | None) -> ListNode | None:
    """
     21. Merge Two Sorted Lists https://leetcode.com/problems/merge-two-sorted-lists/

    Examples:
    >>> listnode_to_list(p21(list_to_listnode([1, 2, 4]), list_to_listnode([1, 3, 4])))
    [1, 1, 2, 3, 4, 4]
    >>> listnode_to_list(p21(list_to_listnode([]), list_to_listnode([])))
    []
    >>> listnode_to_list(p21(list_to_listnode([]), list_to_listnode([0])))
    [0]
    """
    head = pointer = ListNode()

    while list1 and list2:
        if list1.val < list2.val:
            pointer.next = list1
            list1 = list1.next
        else:
            pointer.next = list2
            list2 = list2.next
        pointer = pointer.next

    if list1:
        pointer.next = list1

    if list2:
        pointer.next = list2

    return head.next


def p22(n: int) -> list[str]:
    """
    22. Generate Parentheses https://leetcode.com/problems/generate-parentheses/

    Examples:
    >>> p22(3)
    ['((()))', '(()())', '(())()', '()(())', '()()()']
    >>> p22(1)
    ['()']
    """
    if n == 1:
        return ["()"]

    res: list[str] = [""]
    for _ in range(2 * n):
        temp = []
        for x in res:
            if x.count("(") < n:
                temp.append(x + "(")
            if x.count("(") > x.count(")"):
                temp.append(x + ")")
        res = temp

    return res


def p23(lists: list[ListNode | None]) -> ListNode | None:
    """
    23. Merge k Sorted Lists https://leetcode.com/problems/merge-k-sorted-lists/

    Lessons learned:
    - Sometimes the hard problems aren't that hard.

    Examples:
    >>> listnode_to_list(p23([list_to_listnode([1, 4, 5]), list_to_listnode([1, 3, 4]), list_to_listnode([2, 6])]))
    [1, 1, 2, 3, 4, 4, 5, 6]
    >>> listnode_to_list(p23([]))
    []
    >>> listnode_to_list(p23([list_to_listnode([])]))
    []
    """
    head = pointer = ListNode()

    while any(x for x in lists):
        min_val = float("inf")
        min_idx = -1
        for i, x in enumerate(lists):
            if x and x.val < min_val:
                min_val = x.val
                min_idx = i

        pointer.next = lists[min_idx]
        lists[min_idx] = lists[min_idx].next
        pointer = pointer.next

    return head.next


def p23_2(lists: list[ListNode | None]) -> ListNode | None:
    """
    Examples:
    >>> listnode_to_list(p23_2([list_to_listnode([1, 4, 5]), list_to_listnode([1, 3, 4]), list_to_listnode([2, 6])]))
    [1, 1, 2, 3, 4, 4, 5, 6]
    >>> listnode_to_list(p23_2([]))
    []
    >>> listnode_to_list(p23_2([list_to_listnode([])]))
    []
    """
    if not lists:
        return None

    while len(lists) > 1:
        lists.append(p21(lists.pop(0), lists.pop(0)))

    return lists[0]


def p24(head: ListNode | None) -> ListNode | None:
    """
    24. Swap Nodes in Pairs https://leetcode.com/problems/swap-nodes-in-pairs/

    Examples:
    >>> listnode_to_list(p24(list_to_listnode([])))
    []
    >>> listnode_to_list(p24(list_to_listnode([1])))
    [1]
    >>> listnode_to_list(p24(list_to_listnode([1, 2])))
    [2, 1]
    >>> listnode_to_list(p24(list_to_listnode([1, 2, 3])))
    [2, 1, 3]
    >>> listnode_to_list(p24(list_to_listnode([1, 2, 3, 4])))
    [2, 1, 4, 3]
    >>> listnode_to_list(p24(list_to_listnode([1, 2, 3, 4, 5])))
    [2, 1, 4, 3, 5]
    """
    if not head:
        return None
    if not head.next:
        return head

    pointer = head
    new_head = head.next

    # 1 2 3 4
    # 2 1 4 3
    while pointer and pointer.next:
        one = pointer
        two = pointer.next
        three = pointer.next.next
        four = pointer.next.next.next if three else None

        one.next = four if four else three
        two.next = one

        pointer = three

    return new_head


def p26(nums: list[int]) -> int:
    """
    26. Remove Duplicates from Sorted Array https://leetcode.com/problems/remove-duplicates-from-sorted-array/

    Examples:
    >>> p26([1, 1, 2])
    2
    >>> p26([0,0,1,1,1,2,2,3,3,4])
    5
    """
    k = 0
    for i in range(1, len(nums)):
        if nums[k] != nums[i]:
            k += 1
            nums[k] = nums[i]
    return k + 1


def p36(board: list[list[str]]) -> bool:
    """
    36. Valid Sudoku https://leetcode.com/problems/valid-sudoku/

    Examples:
    >>> board = [
    ...     ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ...     ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    ...     [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ...     ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ...     ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ...     ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    ...     [".", "6", ".", ".", ".", ".", "2", "8", "."],
    ...     [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    ...     [".", ".", ".", ".", "8", ".", ".", "7", "9"],
    ... ]
    >>> p36(board)
    True
    >>> board = [
    ...     ["8", "3", ".", ".", "7", ".", ".", ".", "."],
    ...     ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    ...     [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ...     ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ...     ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ...     ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    ...     [".", "6", ".", ".", ".", ".", "2", "8", "."],
    ...     [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    ...     [".", ".", ".", ".", "8", ".", ".", "7", "9"],
    ... ]
    >>> p36(board)
    False
    """
    mat = np.char.replace(np.array(board), ".", "0").astype(int)

    for i in range(mat.shape[0]):
        if not (np.bincount(mat[i, :])[1:] <= 1).all():
            return False
        if not (np.bincount(mat[:, i])[1:] <= 1).all():
            return False

    for i in range(3):
        for j in range(3):
            if not (
                np.bincount(mat[3 * i : 3 * i + 3, 3 * j : 3 * j + 3].flatten())[1:] <= 1
            ).all():
                return False

    return True


def p45(nums: list[int]) -> int:
    """
    45. Jump Game II https://leetcode.com/problems/jump-game-ii/

    Lessons learned:
    - This is the backward version of the dynamic programming solution from
    problem 55 Jump Game, except here we keep track of the move counts.
    - It turns out that the greedy solution is optimal. The intuition is that we
    always want to jump to the farthest reachable index. The proof is by
    contradiction. Suppose we have a better solution that jumps to a closer
    index. Then we can always replace that jump with a jump to the farthest
    reachable index, and the new solution will be at least as good as the
    original one. The only necessary jumps are the ones that allow a new
    farthest index to be reached.

    Examples:
    >>> p45([2,3,1,1,4])
    2
    >>> p45([2,3,0,1,4])
    2
    """
    n = len(nums)
    reachable = [0] * (n - 1) + [1]
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, min(n, i + nums[i] + 1)):
            if reachable[j]:
                reachable[i] = (
                    min(1 + reachable[j], reachable[i]) if reachable[i] != 0 else 1 + reachable[j]
                )

    return reachable[0] - 1


def p45_2(nums: list[int]) -> int:
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


def p49(strs: list[str]) -> list[list[str]]:
    """
    49. Group Anagrams https://leetcode.com/problems/group-anagrams/

    Examples:
    >>> p49(["eat","tea","tan","ate","nat","bat"])
    [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
    >>> p49([""])
    [['']]
    >>> p49(["a"])
    [['a']]
    """

    def group_key(s: str) -> tuple[str, ...]:
        return tuple(sorted(s))

    groups = defaultdict(list)
    for s in strs:
        groups[group_key(s)].append(s)

    return list(groups.values())


def p54(matrix: list[list[int]]) -> list[int]:
    """
    54. Spiral Matrix https://leetcode.com/problems/spiral-matrix/

    Examples:
    >>> p54([[1,2,3],[4,5,6],[7,8,9]])
    [1, 2, 3, 6, 9, 8, 7, 4, 5]
    >>> p54([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
    >>> p54([[1]])
    [1]
    >>> p54([[1,2],[3,4]])
    [1, 2, 4, 3]
    """
    m, n = len(matrix), len(matrix[0])
    directions = cycle([[0, 1], [1, 0], [0, -1], [-1, 0]])
    cur_pos, steps = (0, 0), 1
    visited, elements = set(), [matrix[0][0]]
    visited.add(cur_pos)
    for dm, dn in directions:
        if steps == m * n:
            break

        while True:
            new_pos = (cur_pos[0] + dm, cur_pos[1] + dn)

            if new_pos in visited or not (0 <= new_pos[0] < m and 0 <= new_pos[1] < n):
                break

            cur_pos = new_pos
            elements.append(matrix[cur_pos[0]][cur_pos[1]])
            visited.add(cur_pos)
            steps += 1

    return elements


def p55(nums: list[int]) -> bool:
    """
    55. Jump Game https://leetcode.com/problems/jump-game/

    Lessons learned:
    - The forward version of the dynamic programming solution is more intuitive,
    but it is slow. The backward version is much faster.
    - The second version is even better, avoiding the second for loop. The
    intuition there is that we only need to keep track of the minimum index that
    can reach the end.

    Examples:
    >>> p55([2,3,1,1,4])
    True
    >>> p55([3,2,1,0,4])
    False
    >>> p55(list(range(10, -1, -1)) + [0])
    False
    """
    n = len(nums)
    reachable = [0] * (n - 1) + [1]
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, min(n, i + nums[i] + 1)):
            if reachable[j]:
                reachable[i] = 1
                break

    return reachable[0] == 1


def p55_2(nums: list[int]) -> bool:
    n = len(nums)
    current = n - 1
    for i in range(n - 2, -1, -1):
        step = nums[i]

        if i + step >= current:
            current = i

    return current == 0


def get_fibonacci_number(n: int) -> int:
    """Return the nth Fibonacci number, where F(0) = 0 and F(1) = 1.

    Examples:
    >>> [get_fibonacci_number(i) for i in range(10)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
    phi = (1 + 5**0.5) / 2
    return int(phi**n / 5**0.5 + 0.5)


def get_binary_strings_no_consecutive_zeros(n: int) -> list[str]:
    """
    Examples:
    >>> get_binary_strings_no_consecutive_zeros(1)
    ['0', '1']
    >>> get_binary_strings_no_consecutive_zeros(2)
    ['01', '11', '10']
    >>> get_binary_strings_no_consecutive_zeros(3)
    ['011', '111', '101', '010', '110']
    >>> [len(get_binary_strings_no_consecutive_zeros(i)) for i in range(1, 10)]
    [2, 3, 5, 8, 13, 21, 34, 55, 89]
    """
    if n == 1:
        return ["0", "1"]

    prev = get_binary_strings_no_consecutive_zeros(n - 1)
    return [x + "1" for x in prev] + [x + "0" for x in prev if not x.endswith("0")]


def p91(s: str) -> int:
    """
    91. Decode Ways https://leetcode.com/problems/decode-ways/

    Lessons learned:
    - My first observation was that characters in "34567890" acted as separators,
    where I could split the string into substrings that could be decoded
    independently. My second observation was that the number of ways to decode a
    string of nothing but "1" and "2" characters was the Fibonacci number
    F(n+1), where n is the number of characters in the string. Combining these
    two speedups with recursion gave me the first solution, which had middle of
    the pack runtime and memory usage.
    - The second solution is a very clean dynamic programming approach I lifted
    from the discussion section. Define

        dp(i) = number of ways to decode the substring s[:i]

    The recurrence relation is

        dp(i) = dp(i-1) + dp(i-2) if "11" <= s[i-2:i] <= "26" and s[i-2:i] != "20"
                = dp(i-1) if s[i-1] != "0" and s[i-2:i] > "26"
                = dp(i-2) if "10" == s[i-2:i] or "20" == s[i-2:i]
                = 0 otherwise
        dp(0) = 1
        dp(1) = 1 if s[0] != "0" else 0

    - Fun fact: the number of binary strings of length n with no consecutive zeros
    corresponds to the Fibonacci number F(n+2). This diagram helps visualize the
    recursion:
    https://en.wikipedia.org/wiki/Composition_(combinatorics)#/media/File:Fibonacci_climbing_stairs.svg.

    Examples:
    >>> p91("0")
    0
    >>> p91("06")
    0
    >>> p91("1")
    1
    >>> p91("12")
    2
    >>> p91("111")
    3
    >>> p91("35")
    1
    >>> p91("226")
    3
    >>> p91("2020")
    1
    >>> p91("2021")
    2
    >>> p91("2022322")
    6
    """
    valid_codes = {str(x) for x in range(1, 27)}

    def recurse(s1: str) -> int:
        if s1[0] == "0":
            return 0

        if len(s1) == 1:
            return 1

        if len(s1) == 2:
            if int(s1) <= 26 and s1[1] != "0":
                return 2
            if int(s1) <= 26 and s1[1] == "0":
                return 1
            if int(s1) > 26 and s1[1] != "0":
                return 1
            return 0

        if set(s1) <= {"1", "2"}:
            return get_fibonacci_number(len(s1) + 1)

        if s1[:2] in valid_codes:
            return recurse(s1[1:]) + recurse(s1[2:])

        return recurse(s1[1:])

    total = 1
    prev = 0
    for i, c in enumerate(s):
        if c in "34567890":
            total *= recurse(s[prev : i + 1])
            prev = i + 1

    if s[prev:]:
        total *= recurse(s[prev:])

    return total


def p91_2(s: str) -> int:
    """
    Examples:
    >>> p91_2("0")
    0
    >>> p91_2("06")
    0
    >>> p91_2("1")
    1
    >>> p91_2("12")
    2
    >>> p91_2("111")
    3
    >>> p91_2("35")
    1
    >>> p91_2("226")
    3
    >>> p91_2("2020")
    1
    >>> p91_2("2021")
    2
    >>> p91_2("2022322")
    6
    """
    if not s or s[0] == "0":
        return 0

    dp = [0] * (len(s) + 1)
    dp[0:2] = [1, 1]

    for i in range(2, len(s) + 1):
        if "11" <= s[i - 2 : i] <= "19" or "21" <= s[i - 2 : i] <= "26":
            dp[i] = dp[i - 1] + dp[i - 2]
        elif s[i - 1] != "0":
            dp[i] = dp[i - 1]
        elif "10" == s[i - 2 : i] or "20" == s[i - 2 : i]:
            dp[i] = dp[i - 2]

    return dp[-1]
