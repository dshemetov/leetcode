"""
Python solutions to LeetCode problems.

Keeping it as a monofile because there's something funny about it.
"""

import math
from array import array
from bisect import bisect_left, insort
from collections import Counter, defaultdict, deque, namedtuple
from collections.abc import Generator
from fractions import Fraction
from itertools import cycle, product
from queue import PriorityQueue
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np


def p1(nums: list[int], target: int) -> list[int]:
    """
    1. Two Sum https://leetcode.com/problems/two-sum/

    Examples:
    >>> p1([3, 3], 6)
    [0, 1]
    >>> p1([3, 2, 4], 7)
    [0, 2]
    """
    ix_map = defaultdict(list)
    for i, x in enumerate(nums):
        ix_map[x].append(i)

    for x in ix_map:
        if ix_map.get(target - x):
            if x == target - x and len(ix_map.get(x)) == 2:
                return ix_map.get(target - x)
            if x != target - x:
                return [ix_map.get(x)[0], ix_map.get(target - x)[0]]
    return 0


class ListNode:
    def __init__(self, val=0, next=None):  # pylint: disable=redefined-builtin
        self.val = val
        self.next = next

    @staticmethod
    def from_list(lst: list[int]) -> Optional["ListNode"]:
        if not lst:
            return None

        original_head = head = ListNode(lst[0])
        for x in lst[1:]:
            head.next = ListNode(x)
            head = head.next

        return original_head

    @staticmethod
    def from_int(i: int) -> "ListNode":
        num_list = None
        for x in str(i):
            num_list = ListNode(val=int(x), next=num_list)
        return num_list


def listnode_to_list(head) -> list[int]:
    """
    Examples:
    >>> listnode_to_list(ListNode.from_list([1, 2, 3, 4, 5]))
    [1, 2, 3, 4, 5]
    """
    lst = []
    ptr = head
    while ptr:
        lst.append(ptr.val)
        ptr = ptr.next

    return lst


def list_to_int(lst: ListNode) -> int:
    """
    Examples:
    >>> list_to_int(ListNode.from_int(0))
    0
    >>> list_to_int(ListNode.from_int(2))
    2
    >>> list_to_int(ListNode.from_int(12))
    12
    >>> list_to_int(ListNode.from_int(15))
    15
    >>> list_to_int(ListNode.from_int(255))
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
    >>> list_to_int(p2(ListNode.from_int(0), ListNode.from_int(15)))
    15
    >>> list_to_int(p2(ListNode.from_int(12), ListNode.from_int(15)))
    27
    >>> list_to_int(p2(ListNode.from_int(12), ListNode.from_int(153)))
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


def get_median_sorted(nums: list[int]) -> float:
    if len(nums) == 1:
        return nums[0]

    mid = len(nums) // 2

    if len(nums) % 2 == 0:
        return (nums[mid] + nums[mid - 1]) / 2

    return nums[mid]


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
    >>> listnode_to_list(p19(ListNode.from_list([1, 2, 3, 4, 5]), 1))
    [1, 2, 3, 4]
    >>> listnode_to_list(p19(ListNode.from_list([1, 2, 3, 4, 5]), 2))
    [1, 2, 3, 5]
    >>> listnode_to_list(p19(ListNode.from_list([1, 2, 3, 4, 5]), 3))
    [1, 2, 4, 5]
    >>> listnode_to_list(p19(ListNode.from_list([1, 2, 3, 4, 5]), 4))
    [1, 3, 4, 5]
    >>> listnode_to_list(p19(ListNode.from_list([1, 2, 3, 4, 5]), 5))
    [2, 3, 4, 5]
    >>> listnode_to_list(p19(ListNode.from_list([1]), 1))
    []
    >>> listnode_to_list(p19(ListNode.from_list([1, 2]), 1))
    [1]
    >>> listnode_to_list(p19(ListNode.from_list([1, 2]), 2))
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
    >>> listnode_to_list(p21(ListNode.from_list([1, 2, 4]), ListNode.from_list([1, 3, 4])))
    [1, 1, 2, 3, 4, 4]
    >>> listnode_to_list(p21(ListNode.from_list([]), ListNode.from_list([])))
    []
    >>> listnode_to_list(p21(ListNode.from_list([]), ListNode.from_list([0])))
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
    >>> listnode_to_list(p23([ListNode.from_list([1, 4, 5]), ListNode.from_list([1, 3, 4]), ListNode.from_list([2, 6])]))
    [1, 1, 2, 3, 4, 4, 5, 6]
    >>> listnode_to_list(p23([]))
    []
    >>> listnode_to_list(p23([ListNode.from_list([])]))
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
    >>> listnode_to_list(p23_2([ListNode.from_list([1, 4, 5]), ListNode.from_list([1, 3, 4]), ListNode.from_list([2, 6])]))
    [1, 1, 2, 3, 4, 4, 5, 6]
    >>> listnode_to_list(p23_2([]))
    []
    >>> listnode_to_list(p23_2([ListNode.from_list([])]))
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
    >>> listnode_to_list(p24(ListNode.from_list([])))
    []
    >>> listnode_to_list(p24(ListNode.from_list([1])))
    [1]
    >>> listnode_to_list(p24(ListNode.from_list([1, 2])))
    [2, 1]
    >>> listnode_to_list(p24(ListNode.from_list([1, 2, 3])))
    [2, 1, 3]
    >>> listnode_to_list(p24(ListNode.from_list([1, 2, 3, 4])))
    [2, 1, 4, 3]
    >>> listnode_to_list(p24(ListNode.from_list([1, 2, 3, 4, 5])))
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


def p25(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    25. Reverse Nodes in k-Group https://leetcode.com/problems/reverse-nodes-in-k-group/description/

    TODO: Implement this.
    """
    ...


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
                np.bincount(mat[3 * i : 3 * i + 3, 3 * j : 3 * j + 3].flatten())[1:]
                <= 1
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
                    min(1 + reachable[j], reachable[i])
                    if reachable[i] != 0
                    else 1 + reachable[j]
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


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def p102(root: TreeNode | None) -> list[list[int]]:
    """
    102. Binary Tree Level Order Traversal https://leetcode.com/problems/binary-tree-level-order-traversal/

    Examples:
    >>> p102(TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7))))
    [[3], [9, 20], [15, 7]]
    >>> p102(TreeNode(1))
    [[1]]
    >>> p102(None)
    []
    """
    if not root:
        return []

    queue = [root]
    result = []
    while queue:
        level = []
        new_queue = []
        for x in queue:
            level.append(x.val)
            if x.left:
                new_queue.append(x.left)
            if x.right:
                new_queue.append(x.right)
        result.append(level)
        queue = new_queue
    return result


class Node:
    def __init__(self, val: int = 0, neighbors: list["Node"] | None = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def adjacency_list_to_node_graph(adjacency_list: list[list[int]]) -> "Node":
    """Build a node-based graph from an adjacency list.

    Examples:
    >>> node_graph_to_adjacency_list(adjacency_list_to_node_graph([[1, 2], [1, 4], [2, 3], [3, 4]]))
    [[1, 2], [1, 4], [2, 3], [3, 4]]
    """
    if adjacency_list == [[]]:
        return Node(1)

    node_index = {}
    for x, y in adjacency_list:
        if (xnode := node_index.get(x)) is None:
            xnode = Node(x)
            node_index[x] = xnode
        if (ynode := node_index.get(y)) is None:
            ynode = Node(y)
            node_index[y] = ynode

        xnode.neighbors.append(ynode)
        ynode.neighbors.append(xnode)

    return node_index[1]


def node_graph_to_adjacency_list(node: "Node") -> "Node":
    """Traverse through a graph and build an adjacency list.

    Examples:
    >>> node_graph_to_adjacency_list(adjacency_list_to_node_graph([[1, 2], [1, 4], [2, 3], [3, 4]]))
    [[1, 2], [1, 4], [2, 3], [3, 4]]
    """
    adjacency_list = set()
    visited = set()
    node_queue = deque([node])

    while node_queue:
        node = node_queue.popleft()
        visited.add(node.val)

        for neighbor in node.neighbors:
            adjacency_list.add(tuple(sorted([node.val, neighbor.val])))

            if neighbor.val not in visited:
                node_queue.append(neighbor)

    return sorted([list(e) for e in adjacency_list], key=lambda x: (x[0], x[1]))


def p133(node: "Node") -> "Node":
    """
    133. Clone Graph https://leetcode.com/problems/clone-graph/

    Examples:
    >>> p133(None)
    >>> node_graph_to_adjacency_list(p133(adjacency_list_to_node_graph([[1, 2], [1, 4], [2, 3], [3, 4]])))
    [[1, 2], [1, 4], [2, 3], [3, 4]]
    """
    if node is None:
        return None

    node_queue = deque([node])
    clone_index = {node.val: Node(node.val)}
    while node_queue:
        cur_node = node_queue.popleft()
        cur_clone = clone_index[cur_node.val]

        for neighbor in cur_node.neighbors:
            if neighbor.val not in clone_index:
                clone_index[neighbor.val] = Node(neighbor.val)
                node_queue.append(neighbor)

            cur_clone.neighbors.append(clone_index[neighbor.val])

    return clone_index[1]


def p141(head: ListNode | None) -> bool:
    """
    141. Linked List Cycle https://leetcode.com/problems/linked-list-cycle/

    Lessons learned:
    - We use the classic two-pointer cycle detection algorithm known as Floyd's
    Tortoise and Hare.
    - One intuitive way to think about why this works is to consider the
    tortoise as being ahead of the hare, once the tortoise is in the cycle, and
    the hare gets closer to the tortoise with each step. Eventually, the hare
    will catch up to the tortoise, and we will have a cycle.
    - An algebraic proof: we want to show that

        i  = n + m
        2i = n + k * p + m
        ------------------
        n + m = k * p

    has a solution in k >= 0 and m >= 0, where n is the number of nodes before
    the cycle, p is the number of nodes in the cycle, and m is the number of
    nodes in the cycle before the hare catches up to the tortoise. Setting k = n
    and m = n * p - n, we have a solution. For instance, if n = 5 and p = 3,
    this gives the solution k = 5 and m = 10, which translates to i = 15 and 2i
    = 30, which works in the table below:

        Tortoise index: 0, 1, 2, 3, 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, ...
        Tortoise node:  1, 2, 3, 4, 5,  6,  7,  5,  6,  7,  5,  6,  7,  5,  6,  7, ...
        Hare index:     0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, ...
        Hare node:      1, 3, 5, 7, 5,  7,  5,  7,  5,  7,  5,  7,  5,  7,  5,  7, ...

    This is not the earliest solution, but it is an existence proof.

    Examples:
    >>> p141(ListNode.from_list([3, 2, 0, -4], 1))
    True
    >>> p141(ListNode.from_list([1, 2], 0))
    True
    >>> p141(ListNode.from_list([1], -1))
    False
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def p151(s: str) -> str:
    """
    151. Reverse Words In A String https://leetcode.com/problems/reverse-words-in-a-string/

    Lesson learned:
    - Python string built-ins are fast.
    - The follow-up asks: can you use only O(1) extra space? Here is the trick:
    first reverse the whole string and then reverse each word. Reversing each
    requires keeping track of two pointers: the start of a word and the end of a
    word (terminated by a space).

    Examples:
    >>> p151("the sky is blue")
    'blue is sky the'
    >>> p151("  hello world!  ")
    'world! hello'
    >>> p151("a good   example")
    'example good a'
    >>> p151("  Bob    Loves  Alice   ")
    'Alice Loves Bob'
    """
    return " ".join(s.split()[::-1])


def p151_2(s: str) -> str:
    """
    Examples:
    >>> p151_2("the sky is blue")
    'blue is sky the'
    >>> p151_2("  hello world!  ")
    'world! hello'
    >>> p151_2("a good   example")
    'example good a'
    >>> p151_2("  Bob    Loves  Alice   ")
    'Alice Loves Bob'
    """
    a = array("u", [])
    a.fromunicode(s.strip())
    a.reverse()

    # Reverse each word
    n = len(a)
    lo = 0
    for i in range(n):
        if a[i] == " ":
            hi = i - 1
            while lo < hi:
                a[lo], a[hi] = a[hi], a[lo]
                lo += 1
                hi -= 1
            lo = i + 1

    hi = n - 1
    while lo < hi:
        a[lo], a[hi] = a[hi], a[lo]
        lo += 1
        hi -= 1

    # Contract spaces in the string
    lo, space = 0, 0
    for i in range(n):
        space = space + 1 if a[i] == " " else 0
        if space <= 1:
            a[lo] = a[i]
            lo += 1

    return "".join(a[:lo])


def p167(numbers: list[int], target: int) -> list[int]:
    """
    167. Two Sum II - Input Array Is Sorted https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/

    Examples:
    >>> p167([2,7,11,15], 9)
    [1, 2]
    >>> p167([2,3,4], 6)
    [1, 3]
    >>> p167([-1,0], -1)
    [1, 2]
    """
    lo, hi = 0, len(numbers) - 1
    while lo < hi:
        s = numbers[lo] + numbers[hi]
        if s < target:
            lo += 1
        elif s > target:
            hi -= 1
        else:
            return [lo + 1, hi + 1]
    return 0


def p200(grid: list[list[str]]) -> int:
    """
    200. Number of Islands https://leetcode.com/problems/number-of-islands/
    """
    n, m = len(grid), len(grid[0])
    visited = set()

    def dfs(i: int, j: int):
        unexplored = {(i, j)}
        while unexplored:
            i_, j_ = unexplored.pop()
            visited.add((i_, j_))

            for i__, j__ in [(i_ + 1, j_), (i_ - 1, j_), (i_, j_ + 1), (i_, j_ - 1)]:
                if (
                    0 <= i__ < n
                    and 0 <= j__ < m
                    and (i__, j__) not in visited
                    and grid[i__][j__] == "1"
                ):
                    unexplored.add((i__, j__))

    islands = 0
    for i in range(n):
        for j in range(m):
            if (i, j) not in visited and grid[i][j] == "1":
                dfs(i, j)
                islands += 1

    return islands


def p212(board: list[list[str]], words: list[str]) -> list[str]:
    """
    212. Word Search II https://leetcode.com/problems/word-search-ii/

    Examples:
    >>> set(p212([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], ["oath","pea","eat","rain"])) == set(["eat", "oath"])
    True
    >>> p212([["a","b"],["c","d"]], ["abcb"])
    []
    >>> board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
    >>> words = ["oath","pea","eat","rain", "oat", "oatht", "naaoetaerkhi", "naaoetaerkhii"]
    >>> set(p212(board, words)) == set(["eat", "oath", "oat", "naaoetaerkhi"])
    True
    """

    class Trie:
        def __init__(self):
            self.root = {}

        def insert(self, word: str) -> None:
            node = self.root
            for char in word:
                node = node.setdefault(char, {})
            node["#"] = "#"

        def remove(self, word: str) -> None:
            node = self.root
            path = []
            for char in word:
                path.append((node, char))
                node = node[char]
            node.pop("#")
            for node, char in reversed(path):
                if not node[char]:
                    node.pop(char)
                else:
                    break

    def dfs(
        i: int,
        j: int,
        node: dict,
        path: str,
        board: list[list[str]],
        found_words: set[str],
    ) -> None:
        if node.get("#"):
            found_words.add(path)
            trie.remove(path)

        board[i][j] = "$"

        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ni, nj = (i + di, j + dj)
            if (
                0 <= ni < len(board)
                and 0 <= nj < len(board[0])
                and board[ni][nj] in node
                and len(path) < 12
            ):
                dfs(
                    ni,
                    nj,
                    node[board[ni][nj]],
                    path + board[ni][nj],
                    board,
                    found_words,
                )

        board[i][j] = path[-1]

    def filter_words(words: list[str]) -> list[str]:
        board_chars = set()
        for row in board:
            board_chars |= set(row)
        return [word for word in words if set(word) <= board_chars]

    words = filter_words(words)

    trie = Trie()
    for word in words:
        trie.insert(word)

    n, m = len(board), len(board[0])
    found_words = set()
    for i in range(n):
        for j in range(m):
            if board[i][j] in trie.root:
                dfs(i, j, trie.root[board[i][j]], board[i][j], board, found_words)

    return list(found_words)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def make_binary_tree(lst: list[int]) -> TreeNode:
    """Fills a binary tree from left to right."""
    if not lst:
        return None
    root = TreeNode(lst[0])
    queue = [root]
    i = 1
    while i < len(lst):
        node = queue.pop(0)
        if lst[i] is not None:
            node.left = TreeNode(lst[i])
            queue.append(node.left)
        i += 1
        if i < len(lst) and lst[i] is not None:
            node.right = TreeNode(lst[i])
            queue.append(node.right)
        i += 1
    return root


def p222(root: TreeNode | None) -> int:
    """
    222. Count Complete Tree Nodes https://leetcode.com/problems/count-complete-tree-nodes/

    Lessons learned:
    - A complete binary tree is a binary tree in which every level is completely
    filled, except for the last where the nodes must be as far left as possible.

    Examples:
    >>> p222(make_binary_tree([1,2,3,4,5,6]))
    6
    >>> p222(make_binary_tree([1,2,3,4,5,6,None]))
    6
    >>> p222(make_binary_tree([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]))
    15
    >>> p222(make_binary_tree([1,2,3,4,5,6,7,8,9,10,11,12,None,None,None]))
    12
    """
    if not root:
        return 0

    height = -1
    node = root
    while node:
        height += 1
        node = node.left

    if height == 0:
        return 1

    def is_node_in_tree(root: TreeNode, i: int) -> bool:
        node = root
        for c in format(i, f"0{height}b"):
            node = node.left if c == "0" else node.right
        return bool(node)

    lo, hi = 0, 2 ** (height) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        node_in_tree = is_node_in_tree(root, mid)
        if node_in_tree:
            lo = mid + 1
        else:
            hi = mid - 1

    return 2 ** (height) + lo - 1


def p223(
    ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int
) -> int:
    """
    223. Rectangle Area https://leetcode.com/problems/rectangle-area/
    """
    A1 = (ax2 - ax1) * (ay2 - ay1)
    A2 = (bx2 - bx1) * (by2 - by1)
    I = max(min(ax2, bx2) - max(ax1, bx1), 0) * max(min(ay2, by2) - max(ay1, by1), 0)
    return A1 + A2 - I


def p242(s: str, t: str) -> bool:
    """
    242. Valid Anagram https://leetcode.com/problems/valid-anagram/
    """
    return Counter(s) == Counter(t)


def p258(num: int) -> int:
    """
    258. Add Digits https://leetcode.com/problems/add-digits/

    Lessons learned:
    - Turns out this can be solved with modular arithmetic because 10 ** n == 1 mod 9

    Examples:
    >>> p258(38)
    2
    >>> p258(0)
    0
    """
    if num == 0:
        return num
    if num % 9 == 0:
        return 9
    return num % 9


def p263(n: int) -> bool:
    """
    263. Ugly Number https://leetcode.com/problems/ugly-number/
    """
    if n < 1:
        return False
    while n % 2 == 0:
        n /= 2
    while n % 3 == 0:
        n /= 3
    while n % 5 == 0:
        n /= 5
    return n == 1


class p295:
    """
    295. Find Median From Data Stream https://leetcode.com/problems/find-median-from-data-stream/

    Examples:
    >>> mf = p295()
    >>> mf.addNum(1)
    >>> mf.addNum(2)
    >>> mf.findMedian()
    1.5
    >>> mf.addNum(3)
    >>> mf.findMedian()
    2.0
    >>> mf = p295()
    >>> mf.addNum(1)
    >>> mf.addNum(2)
    >>> mf.addNum(3)
    >>> mf.addNum(4)
    >>> mf.addNum(5)
    >>> mf.addNum(6)
    >>> mf.addNum(7)
    >>> mf.findMedian()
    4.0
    >>> mf = p295()
    >>> mf.addNum(-1)
    >>> mf.addNum(-2)
    >>> mf.addNum(-3)
    >>> mf.heap
    [-3, -2, -1]
    >>> mf.findMedian()
    -2.0
    """

    def __init__(self):
        self.heap = []

    def addNum(self, num: int) -> None:
        insort(self.heap, num)

    def findMedian(self) -> float:
        if len(self.heap) % 2 == 1:
            return float(self.heap[len(self.heap) // 2])
        return (self.heap[len(self.heap) // 2] + self.heap[len(self.heap) // 2 - 1]) / 2


def p316(s: str) -> str:
    """
    316. Remove Duplicate Letters https://leetcode.com/problems/remove-duplicate-letters/
    1081. https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/

    Lessons learned:
    - In this one, the solution heuristic can be established with a few examples.
    The key is that we can greedily remove left-most duplicated letters that are
    larger than the next letter. For example, if we have cbxxx and we can remove
    c or another letter, then we will have bxxx < cbxx.

    Examples:
    >>> p316("bcabc")
    'abc'
    >>> p316("cbacdcbc")
    'acdb'
    >>> p316("bbcaac")
    'bac'
    >>> p316("bcba")
    'bca'
    """
    letter_counts = Counter(s)
    stack = []
    for c in s:
        letter_counts[c] -= 1
        if c in stack:
            continue
        while stack and c < stack[-1] and letter_counts[stack[-1]] > 0:
            stack.pop()
        stack.append(c)
    return "".join(stack)


def p319(n: int) -> int:
    """
    319. Bulb Switcher https://leetcode.com/problems/bulb-switcher/

    Lessons learned:
    - Testing the array at n=50, I saw that only square numbers remained. From
    there it was easy to prove that square numbers are the only ones with an odd
    number of factors. So this problem is just counting the number of perfect
    squares <= n.

    Examples:
    >>> p319(3)
    1
    >>> p319(0)
    0
    >>> p319(1)
    1
    >>> p319(5)
    2
    """
    arr = np.zeros(n, dtype=int)
    for i in range(1, n + 1):
        for j in range(0, n):
            if (j + 1) % i == 0:
                arr[j] = 1 if arr[j] == 0 else 0
    return sum(arr)


def p319_2(n: int) -> int:
    """
    Examples:
    >>> p319_2(3)
    1
    >>> p319_2(0)
    0
    >>> p319_2(1)
    1
    >>> p319_2(5)
    2
    """
    return int(np.sqrt(n))


def p345(s: str) -> str:
    """
    345. Reverse Vowels of a String https://leetcode.com/problems/reverse-vowels-of-a-string/

    Examples:
    >>> p345("hello")
    'holle'
    >>> p345("leetcode")
    'leotcede'
    """
    if len(s) == 1:
        return s

    hi = len(s) - 1
    s_ = []
    for c in s:
        if c in "aeiouAEIOU":
            while s[hi] not in "aeiouAEIOU":
                hi -= 1
            s_.append(s[hi])
            hi -= 1
        else:
            s_.append(c)

    return "".join(s_)


def p347(nums: list[int], k: int) -> list[int]:
    """
    347. Top K Frequent Elements https://leetcode.com/problems/top-k-frequent-elements/

    Examples:
    >>> p347([1,1,1,2,2,3], 2)
    [1, 2]
    >>> p347([1], 1)
    [1]
    """
    c = Counter(nums)
    return [num for num, _ in c.most_common(k)]


def p349(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    349. Intersection of Two Arrays https://leetcode.com/problems/intersection-of-two-arrays

    Examples:
    >>> p349([1, 2, 2, 1], [2, 2])
    [2]
    >>> p349([4, 9, 5], [9, 4, 9, 8, 4])
    [9, 4]
    """
    return list(set(nums1) & set(nums2))


def p373(nums1: list[int], nums2: list[int], k: int) -> list[list[int]]:
    """
    373. Find K Pairs with Smallest Sums https://leetcode.com/problems/find-k-pairs-with-smallest-sums/

    TODO

    Examples:
    >>> p373([1,7,11], [2,4,6], 3)
    [[1, 2], [1, 4], [1, 6]]
    >>> p373([1,1,2], [1,2,3], 2)
    [[1, 1], [1, 1]]
    >>> p373([1,2], [3], 3)
    [[1, 3], [2, 3]]
    """
    ...


def guess(num: int) -> int:
    __pick__ = 6
    if num == __pick__:
        return 0
    if num > __pick__:
        return -1
    return 1


def p374(n: int) -> int:
    """
    374. Guess Number Higher or Lower https://leetcode.com/problems/guess-number-higher-or-lower/

    Lessons learned:
    - bisect_left has a 'key' argument as of 3.10.

    Examples:
    >>> p374(10)
    6
    """
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        out = guess(mid)
        if out == 1:
            lo = mid + 1
        elif out == -1:
            hi = mid - 1
        else:
            return mid

    return lo


def p374_2(n: int) -> int:
    """
    Examples:
    >>> p374_2(10)
    6
    """
    return bisect_left(range(0, n), 0, lo=0, hi=n, key=lambda x: -guess(x))


def p399(
    equations: list[list[str]], values: list[float], queries: list[list[str]]
) -> list[float]:
    """
    399. Evaluate Division https://leetcode.com/problems/evaluate-division/

    Examples:
    >>> p399([["a","b"],["b","c"]], [2.0,3.0], [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]])
    [6.0, 0.5, -1.0, 1.0, -1.0]
    >>> p399([["a","b"],["b","c"],["bc","cd"]], [1.5,2.5,5.0], [["a","c"],["c","b"],["bc","cd"],["cd","bc"]])
    [3.75, 0.4, 5.0, 0.2]
    >>> p399([["a","b"]], [0.5], [["a","b"],["b","a"],["a","c"],["x","y"]])
    [0.5, 2.0, -1.0, -1.0]
    """
    graph: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), v in zip(equations, values):
        graph[a][b] = v
        graph[b][a] = 1 / v

    def dfs(a: str, b: str) -> float:
        if a not in graph or b not in graph:
            return -1.0
        unexplored = deque([(a, 1.0)])
        visited = set()
        while unexplored:
            node, cost = unexplored.pop()
            visited.add(node)
            if node == b:
                return cost
            for child in graph[node]:
                if child not in visited:
                    unexplored.append((child, cost * graph[node][child]))
        return -1.0

    return [dfs(a, b) for a, b in queries]


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


def ccw(p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int]) -> float:
    """
    Examples:
    >>> ccw((0, 0), (1, 0), (0, 1))
    1.0
    >>> ccw((0, 0), (1, 0), (1, 1))
    1.0
    >>> ccw((0, 0), (1, 0), (1, 0))
    0.0
    >>> ccw((0, 0), (1, 0), (0, -1))
    -1.0
    """
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def polar_angle(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Get the polar angle of the vector from p1 to p2."""
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    return np.arctan2(v1[1], v1[0])


def point_sorter(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Sort by polar angle and break ties by distance."""
    return (polar_angle(p1, p2), -np.linalg.norm((p2[0] - p1[0], p2[1] - p1[1])))


def atan2notan(y: int, x: int) -> Fraction:
    """A polar angle substitute without trigonometry or floating points.

    Imagine tracing out a circle counterclockwise and measuring the angle to the tracing vector
    from the positive x axis. This is the sorted order we wish to achieve. This function will give
    a lexically smaller tuple for smaller angles.
    """
    if x == 0 and y == 0:
        return (0, 0)
    if x > 0 and y >= 0:
        return (0, Fraction(y, x))
    if x == 0 and y > 0:
        return (1, 0)
    if x < 0:
        return (2, Fraction(y, x))
    if x == 0 and y < 0:
        return (3, 0)
    if y < 0 < x:
        return (4, Fraction(y, x))
    raise ValueError("How did you even get here?")


def partition_by(l: list, f: Callable) -> dict:
    """Partition a list into lists based on a predicate."""
    d = defaultdict(list)
    for item in l:
        d[f(item)].append(item)
    return d


def plot_points(points: list[tuple[int, int]], hull: list[tuple[int, int]]):
    x, y = zip(*points)
    plt.scatter(x, y)
    x, y = zip(*hull)
    plt.plot(x, y, color="green")
    plt.show()


def p587(trees: list[list[int]]) -> list[list[int]]:
    """
    587. Erect the Fence https://leetcode.com/problems/erect-the-fence/

    Lessons learned:
    - A broad class of computational geometry algorithms solve this:
    https://en.wikipedia.org/wiki/Convex_hull_algorithms#Algorithms
    - The Graham scan is easy to understand and decently fast:
    https://en.wikipedia.org/wiki/Graham_scan
    - Tip from a graphics guy: avoid representing angles with degrees/radians,
    stay in fractions. This avoids numerical issues with floating points, but
    it's not without its own problems.
    - The atan2 function was invented back in the Fortran days and makes for a
    stable polar angle definition. It's also fast.
    - The edge-cases of the Graham scan are tricky, especially all the cases with
    colinear points.

    Examples:
    >>> p587([[1,1],[2,2],[2,0],[2,4],[3,3],[4,2]])
    [[2, 0], [4, 2], [3, 3], [2, 4], [1, 1]]
    >>> p587([[1,2],[2,2],[4,2]])
    [[1, 2], [2, 2], [4, 2]]
    >>> p587([[1,1],[2,2],[3,3],[2,1],[4,1],[2,3],[1,3]])
    [[1, 1], [2, 1], [4, 1], [3, 3], [2, 3], [1, 3]]
    >>> p587([[3,0],[4,0],[5,0],[6,1],[7,2],[7,3],[7,4],[6,5],[5,5],[4,5],[3,5],[2,5],[1,4],[1,3],[1,2],[2,1],[4,2],[0,3]])
    [[3, 0], [4, 0], [5, 0], [6, 1], [7, 2], [7, 3], [7, 4], [6, 5], [5, 5], [4, 5], [3, 5], [2, 5], [1, 4], [0, 3], [1, 2], [2, 1]]
    >>> p587([[0,2],[0,1],[0,0],[1,0],[2,0],[1,1]])
    [[0, 0], [1, 0], [2, 0], [1, 1], [0, 2], [0, 1]]
    >>> p587([[0,2],[0,4],[0,5],[0,9],[2,1],[2,2],[2,3],[2,5],[3,1],[3,2],[3,6],[3,9],[4,2],[4,5],[5,8],[5,9],[6,3],[7,9],[8,1],[8,2],[8,5],[8,7],[9,0],[9,1],[9,6]])
    [[9, 0], [9, 1], [9, 6], [7, 9], [5, 9], [3, 9], [0, 9], [0, 5], [0, 4], [0, 2], [2, 1]]
    >>> p587([[0,0],[0,1],[0,2],[1,2],[2,2],[3,2],[3,1],[3,0],[2,0],[1,0],[1,1],[3,3]])
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [3, 3], [0, 2], [0, 1]]
    """
    lowest_left_point = (math.inf, math.inf)
    for x, y in trees:
        if y < lowest_left_point[1] or (
            y == lowest_left_point[1] and x < lowest_left_point[0]
        ):
            lowest_left_point = (x, y)

    trees_by_slope = partition_by(
        trees,
        lambda p: atan2notan(p[1] - lowest_left_point[1], p[0] - lowest_left_point[0]),
    )
    slopes = sorted(trees_by_slope.keys())

    # Handles many colinear cases; order doesn't matter for leetcode
    if len(slopes) == 1:
        return trees

    def distance(p1, p2):
        return np.linalg.norm((p1[1] - p2[1], p1[0] - p2[0]))

    # The right-most line should sort by increasing distance from lowest left point
    trees_by_slope[slopes[0]] = sorted(
        trees_by_slope[slopes[0]], key=lambda p: distance(p, lowest_left_point)
    )
    # The left-most line should sort by decreasing distance from lowest left point
    trees_by_slope[slopes[-1]] = sorted(
        trees_by_slope[slopes[-1]], key=lambda p: -distance(p, lowest_left_point)
    )
    # The rest should sort by increasing distance from lowest left point
    for slope in slopes[1:-1]:
        trees_by_slope[slope] = sorted(
            trees_by_slope[slope], key=lambda p: distance(p, lowest_left_point)
        )

    stack = []
    for slope in slopes:
        for tree in trees_by_slope[slope]:
            while len(stack) >= 2 and ccw(stack[-2], stack[-1], tree) < 0:
                stack.pop()
            stack.append(tree)

    return stack


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


def p766(matrix: list[list[int]]) -> bool:
    """
    766. Toeplitz Matrix https://leetcode.com/problems/toeplitz-matrix/

    Examples:
    >>> p766([[1, 2, 3, 4], [5, 1, 2, 3], [9, 5, 1, 2]])
    True
    >>> p766([[1, 2], [2, 2]])
    False
    >>> p766([[11,74,0,93],[40,11,74,7]])
    False
    """
    return all(
        r == 0 or c == 0 or matrix[r - 1][c - 1] == val
        for r, row in enumerate(matrix)
        for c, val in enumerate(row)
    )


def p785(graph: list[list[int]]) -> bool:
    """
    785. Is Graph Bipartite? https://leetcode.com/problems/is-graph-bipartite/

    Lessons learned:
    - A graph is bipartite iff it does not contain any odd cycles. So at first I
    set out to calculate the distances between all nodes and to throw a False if
    I found a loop back to the starting point of odd length. But then I noticed
    that the method I was using was not going to be linear time. I looked up the
    standard method for finding shortest paths between all pairs of nodes in a
    directed, weighted graph (the Floyd-Warshall algorithm), but that was a bit
    overkill too (having a time complexity O(|V|^3)).
    - This problem took over an hour to do. The odd cycles property threw me off,
    making me think that I needed to keep track of node path lengths. Once I let
    go of that idea, I realized that a greedy coloring approach would do the
    trick.

    Examples:
    >>> p785([[1,2,3], [0,2], [0,1,3], [0,2]])
    False
    >>> p785([[1, 3], [0, 2], [1, 3], [0, 2]])
    True
    """
    if not graph:
        return True

    coloring: dict[int, int] = {}

    def dfs(node: int, color: int) -> bool:
        if node in coloring:
            if coloring[node] != color:
                return False
            return True

        coloring[node] = color
        return all(dfs(new_node, color ^ 1) for new_node in graph[node])

    for node in range(len(graph)):
        if node not in coloring and not dfs(node, 0):
            return False

    return True


def p791(order: str, s: str) -> str:
    """
    791. Custom Sort String https://leetcode.com/problems/custom-sort-string/

    Examples:
    >>> p791("cba", "abcd")
    'cbad'
    >>> p791("cba", "abc")
    'cba'
    >>> p791("bcafg", "abcd")
    'bcad'
    """

    def key_fn(t: str) -> int:
        try:
            return order.index(t)
        except ValueError:
            return 30

    return "".join(sorted(s, key=key_fn))


def is_similar(s1: str, s2: str) -> bool:
    """
    Examples:
    >>> is_similar("abc", "abc")
    True
    >>> is_similar("abc", "acb")
    True
    >>> is_similar("abc", "abcd")
    False
    >>> is_similar("abc", "abd")
    False
    """
    if len(s1) != len(s2):
        return False
    diff_chars = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return diff_chars in {0, 2}


def p839(strs: list[str]) -> int:
    """
    839. Similar String Groups https://leetcode.com/problems/similar-string-groups/

    Examples:
    >>> p839(["tars","rats","arts","star"])
    2
    >>> p839(["omv","ovm"])
    1
    >>> p839(["a"])
    1
    >>> p839(["abc","abc"])
    1
    >>> p839(["abc","acb","abc","acb","abc","acb"])
    1
    """
    n = len(strs)
    parent: dict[int, int] = dict({i: i for i in range(n)})

    def find(x: str) -> str:
        y = x
        while True:
            if y != parent[y]:
                y = parent[y]
                continue
            break
        parent[x] = y
        return parent[x]

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if is_similar(strs[i], strs[j]):
                union(i, j)

    return len({find(i) for i in range(n)})


def p876(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    876. Middle of the Linked List https://leetcode.com/problems/middle-of-the-linked-list

    Examples:
    >>> listnode_to_list(p876(ListNode.from_list([1, 2, 3, 4, 5])))
    [3, 4, 5]
    >>> listnode_to_list(p876(ListNode.from_list([1, 2, 3, 4, 5, 6])))
    [4, 5, 6]
    """
    if not head or not head.next:
        return head
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def p899(s: str, k: int) -> str:
    """
    899. Orderly Queue https://leetcode.com/problems/orderly-queue/

    Lessons learned:
    - This problem is such a troll. At first I thought I found a totally
    ridiculous Copilot suggestion, but then I realized that the solution was
    actually dead simple - you can use the rightmost character as a register and
    rotate the string until the correct insertion point.

    Examples:
    >>> p899("cba", 1)
    'acb'
    >>> p899("baaca", 3)
    'aaabc'
    >>> p899("baaca", 1)
    'aacab'
    >>> p899("baaca", 2)
    'aaabc'
    >>> p899("baaca", 4)
    'aaabc'
    >>> p899("badaca", 2)
    'aaabcd'
    >>> p899("badacadeff", 3)
    'aaabcddeff'
    """
    if k == 1:
        return min(s[i:] + s[:i] for i in range(len(s)))

    return "".join(sorted(s))


class p901:
    """
    901. Online Stock Span https://leetcode.com/problems/online-stock-span/

    Lessons learned:
    - This uses a monotonically decreasing stack (MDS) to keep track of the
    previous stock prices and their spans.

    Examples:
    >>> obj = p901()
    >>> obj.next(100)
    1
    >>> obj.next(80)
    1
    >>> obj.next(60)
    1
    >>> obj.next(70)
    2
    >>> obj.next(60)
    1
    >>> obj.next(75)
    4
    >>> obj.next(85)
    6
    """

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        self.stack.append([price, span])
        return span


def p934(grid: list[list[int]]) -> int:
    """
    934. Shortest Bridge https://leetcode.com/problems/shortest-bridge/

    Lessons learned:
    - This problem has a couple sub-problems that allow for variants.
    - First, there is the problem of correctly coloring the connected components.
    This can be done with a simple DFS and an extra coloring dictionary, but
    here we modify the input grid to save space.
    - Second, there is the path-finding problem. This can be done with BFS.

    Examples:
    >>> p934([[0,1],[1,0]])
    1
    >>> p934([[0,1,0],[0,0,0],[0,0,1]])
    2
    >>> p934([[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]])
    1
    """
    n = len(grid)
    island1 = deque()

    def dfs(i: int, j: int, color: int) -> None:
        unexplored = deque([(i, j)])
        while unexplored:
            i_, j_ = unexplored.pop()
            grid[i_][j_] = color
            if color == 2:
                island1.append((i_, j_))

            for x, y in [(i_ + 1, j_), (i_ - 1, j_), (i_, j_ + 1), (i_, j_ - 1)]:
                if 0 <= x < n and 0 <= y < n and grid[x][y] == 1:
                    unexplored.append((x, y))

    color = 2
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j, color)
                color += 1
                break
        if color == 4:
            break

    unexplored = island1
    next_unexplored = deque()
    distance = 0
    while True:
        while unexplored:
            i, j = unexplored.pop()

            for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                if 0 <= x < n and 0 <= y < n and grid[x][y] != 2:
                    if grid[x][y] == 3:
                        return distance
                    grid[x][y] = 2
                    next_unexplored.append((x, y))

        unexplored = next_unexplored
        next_unexplored = deque()
        distance += 1


def p947(stones: list[list[int]]) -> int:
    """
    947. Most Stones Removed With Same Row or Column https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/

    Lessons learned:
    - The key idea is that we can remove all stones in each connected component
    except one. We can use dfs to find the connected components. Fun fact: the
    dfs can avoid recursion by using a stack.

    Examples:
    >>> p947([[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]])
    5
    >>> p947([[0,0],[0,2],[1,1],[2,0],[2,2]])
    3
    >>> p947([[0,0]])
    0
    >>> p947([[0,0],[0,1],[1,1]])
    2
    >>> p947([[0,1],[1,0]])
    0
    """
    rows = defaultdict(list)
    cols = defaultdict(list)

    for i, (r, c) in enumerate(stones):
        rows[r].append(i)
        cols[c].append(i)

    seen = set()

    def dfs(i: int) -> None:
        """dfs without recursion"""
        stack = [i]
        while stack:
            j = stack.pop()
            seen.add(j)
            for k in rows[stones[j][0]] + cols[stones[j][1]]:
                if k not in seen:
                    stack.append(k)

    n_components = 0
    for i in range(len(stones)):
        if i not in seen:
            dfs(i)
            n_components += 1

    return len(stones) - n_components


def p977(nums: list[int]) -> list[int]:
    """
    977. Squares of a Sorted Array https://leetcode.com/problems/squares-of-a-sorted-array/

    Examples:
    >>> p977([-4,-1,0,3,10])
    [0, 1, 9, 16, 100]
    >>> p977([-7,-3,2,3,11])
    [4, 9, 9, 49, 121]
    >>> p977([-5,-3,-2,-1])
    [1, 4, 9, 25]
    """
    l, r = 0, len(nums) - 1
    res = [0] * len(nums)
    i = len(nums) - 1
    while l <= r:
        left, right = nums[l] ** 2, nums[r] ** 2
        if left > right:
            res[i] = left
            l += 1
        else:
            res[i] = right
            r -= 1
        i -= 1

    return res


def p990(equations: list[str]) -> bool:
    """
    990. Satisfiability of Equality Equations https://leetcode.com/problems/satisfiability-of-equality-equations/

    Lessons learned:
    - This was clearly a graph problem underneath, where you need to find the
    connected components given by the equality statements
    - Efficiently calculating the connected components was hard for me though, so
    learning about the disjoint set data structure was key (also referred to as
    union find):
    https://cp-algorithms.com/data_structures/disjoint_set_union.html

    Examples:
    >>> assert p990(["a==b", "b!=a"]) is False
    >>> assert p990(["x==y", "z==w", "y==z", "a==b", "d==e", "f==g", "e==f", "w==x", "c==d", "b==d", "g!=x"]) is True
    >>> assert p990(["x==y", "z==w", "y==z", "a==b", "d==e", "f==g", "e==f", "w==x", "c==d", "b==d", "g!=x", "a==z"]) is False
    >>> assert p990(["x==a", "w==b", "z==c", "a==b", "b==c", "c!=x"]) is False
    >>> assert p990(["a==b", "c==e", "b==c", "a!=e"]) is False
    >>> assert p990(["a==b", "e==c", "c==b", "a!=e"]) is False
    >>> assert p990(["a==b", "e==c", "c==b", "a!=e"]) is False
    >>> assert p990(["a==b", "e==c", "b==c", "a!=e"]) is False
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while True:
            if parent[x] == x:
                return x
            parent[x] = parent[parent[x]]
            x = parent[x]

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for x, eq, _, y in equations:
        if eq == "=":
            parent.setdefault(x, x)
            parent.setdefault(y, y)
            union(x, y)

    for x, eq, _, y in equations:
        if eq == "!":
            if x == y:
                return False
            if find(x) == find(y):
                return False
    return True


def p1035(nums1: list[int], nums2: list[int]) -> int:
    """
    1035. Uncrossed Lines https://leetcode.com/problems/uncrossed-lines/

    Lessons learned:
    - The solution is identical to (1143 Longest Common Subsequence).

    Examples:
    >>> p1035([1,4,2], [1,2,4])
    2
    >>> p1035([2,5,1,2,5], [10,5,2,1,5,2])
    3
    >>> p1035([1,3,7,1,7,5], [1,9,2,5,1])
    2
    """
    dp_ = [[0 for _ in range(len(nums2) + 1)] for _ in range(len(nums1) + 1)]

    for i in range(1, len(nums1) + 1):
        for j in range(1, len(nums2) + 1):
            if nums1[i - 1] == nums2[j - 1]:
                dp_[i][j] = 1 + dp_[i - 1][j - 1]
            else:
                dp_[i][j] = max(dp_[i - 1][j], dp_[i][j - 1])

    return dp_[-1][-1]


def p1046(stones: list[int]) -> int:
    """
    1046. Last Stone Weight https://leetcode.com/problems/last-stone-weight/

    Examples:
    >>> p1046([2,7,4,1,8,1])
    1
    >>> p1046([1,3])
    2
    >>> p1046([1])
    1
    """
    sorted_stones = sorted(stones)
    while len(sorted_stones) > 1:
        a, b = sorted_stones.pop(), sorted_stones.pop()
        if a != b:
            insort(sorted_stones, a - b)
    return sorted_stones[0] if sorted_stones else 0


def p1047(s: str) -> str:
    """
    1047. Remove All Adjacent Duplicates in String https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/

    Examples:
    >>> p1047("abbaca")
    'ca'
    >>> p1047("aaaaaaaa")
    ''
    """
    stack = []
    for c in s:
        if stack and c == stack[-1]:
            stack.pop()
        else:
            stack.append(c)
    return "".join(stack)


def p1143(text1: str, text2: str) -> int:
    """
    1143. Longest Common Subsequence https://leetcode.com/problems/longest-common-subsequence/

    Lessons learned:
    - This is a classic dynamic programming problem. Define

        dp(i, j) = length of longest common subsequence of text1[:i] and text2[:j]

    The recursion is:

        dp(i, j) = 1 + dp(i - 1, j - 1) if text1[i] == text2[j]
        dp(i, j) = max(dp(i - 1, j), dp(i, j - 1)) otherwise
        dp(i, j) = 0 if i == 0 or j == 0

    - To avoid recursion, we can use a bottom-up approach, where we start with the
    smallest subproblems and build up to the largest, storing the results in a
    table.

    Examples:
    >>> p1143("abcde", "ace")
    3
    >>> p1143("abc", "abc")
    3
    >>> p1143("abc", "def")
    0
    """
    dp_ = [[0 for _ in range(len(text2) + 1)] for _ in range(len(text1) + 1)]

    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            if text1[i - 1] == text2[j - 1]:
                dp_[i][j] = 1 + dp_[i - 1][j - 1]
            else:
                dp_[i][j] = max(dp_[i - 1][j], dp_[i][j - 1])

    return dp_[-1][-1]


def p1171(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    1171. Remove Zero Sum Consecutive Node from Linked List https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/

    Lessons learned:
    - The key here is to use a prefix sum map and make two passes through the
    list. The first pass builds the prefix sum map and in the second pass, if we
    find a prefix sum that we've seen before, we can remove the nodes between
    the two occurrences.

    Examples:
    >>> listnode_to_list(p1171(ListNode.from_list([1,2,-3,3,1])))
    [3, 1]
    >>> listnode_to_list(p1171(ListNode.from_list([1,2,3,-3,4])))
    [1, 2, 4]
    >>> listnode_to_list(p1171(ListNode.from_list([1,2,3,-3,-2])))
    [1]
    >>> listnode_to_list(p1171(ListNode.from_list([5,-3,-4,1,6,-2,-5])))
    [5, -2, -5]
    """
    start = ListNode(0, head)
    prefix_sum_to_node = {0: start}

    current = start
    prefix_sum = 0
    while current:
        prefix_sum += current.val
        prefix_sum_to_node[prefix_sum] = current
        current = current.next

    current = start
    prefix_sum = 0
    while current:
        prefix_sum += current.val
        current.next = prefix_sum_to_node[prefix_sum].next
        current = current.next

    return start.next


def p1293(grid: list[list[int]], k: int) -> int:
    """
    1293. Shortest Path in a Grid With Obstacles Elimination https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/

    Lessons learned:
    - You don't need a dictionary of best distances, just a set of visited nodes
    (since any first visit to a node is the best).
    - You don't need a priority queue, just a queue.

    Examples:
    >>> p1293([[0,0,0],[1,1,0],[0,0,0],[0,1,1],[0,0,0]], 1)
    6
    >>> p1293([[0,1,1],[1,1,1],[1,0,0]], 1)
    -1
    >>> grid = [
    ...     [0,0,0,0,0,0,0,0,0,0],[0,1,1,1,1,1,1,1,1,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,1,1,1,1,1,1,1],[0,1,0,0,0,0,0,0,0,0],[0,1,1,1,1,1,1,1,1,0],
    ...     [0,1,0,0,0,0,0,0,0,0],[0,1,0,1,1,1,1,1,1,1],[0,1,0,1,1,1,1,0,0,0],[0,1,0,0,0,0,0,0,1,0],[0,1,1,1,1,1,1,0,1,0],[0,0,0,0,0,0,0,0,1,0]
    ... ]
    >>> p1293(grid, 1)
    20
    """
    State = namedtuple("State", "steps k i j")
    m, n = len(grid), len(grid[0])

    # Trivial solution: just pick a random Manhattan distance and blow everything up.
    if k >= m + n - 2:
        return m + n - 2

    def get_valid_neighbor_states(s: State) -> Generator[State]:
        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            i, j = s.i + di, s.j + dj
            if 0 <= i < m and 0 <= j < n:
                if grid[i][j] == 0:
                    yield State(s.steps + 1, s.k, i, j)
                elif s.k > 0:
                    yield State(s.steps + 1, s.k - 1, i, j)

    # Don't need a priority queue, since we're only ever visiting each node once.
    # The states will naturally be ordered by steps.
    queue = deque([State(0, k, 0, 0)])
    # We can just use a set instead of a dict, since any first visit to a state has minimum steps.
    seen = {(0, 0, k)}

    while queue:
        current_state = queue.popleft()

        if (current_state.i, current_state.j) == (m - 1, n - 1):
            return current_state.steps

        for state in get_valid_neighbor_states(current_state):
            if (state.i, state.j, state.k) not in seen:
                seen.add((state.i, state.j, state.k))
                queue.append(state)

    return -1


def p1306(arr: list[int], start: int) -> bool:
    """
    1306. Jump Game III https://leetcode.com/problems/jump-game-iii/

    Examples:
    >>> p1306([4,2,3,0,3,1,2], 5)
    True
    >>> p1306([4,2,3,0,3,1,2], 0)
    True
    >>> p1306([3,0,2,1,2], 2)
    False
    """
    seen = set()
    stack = {start}
    while stack:
        ix = stack.pop()

        if arr[ix] == 0:
            return True

        seen.add(ix)

        for ix_ in [ix + arr[ix], ix - arr[ix]]:
            if 0 <= ix_ < len(arr) and ix_ not in seen:
                stack.add(ix_)

    return False


def p1323(num: int) -> int:
    """
    1323. Maximum 69 Number https://leetcode.com/problems/maximum-69-number/

    Lessons learned:
    - Converting to a string and using replace is surprisingly fast.
    - Just need to accept that Python string built-ins are in C-land.

    Examples:
    >>> p1323(9669)
    9969
    >>> p1323(9996)
    9999
    >>> p1323(9999)
    9999
    """
    for i in range(math.floor(math.log10(num)) + 1, -1, -1):
        if num // 10**i % 10 == 6:
            return num + 3 * 10**i
    return num


def p1323_2(num: int) -> int:
    """
    Examples:
    >>> p1323_2(9669)
    9969
    >>> p1323_2(9996)
    9999
    >>> p1323_2(9999)
    9999
    """
    return int(str(num).replace("6", "9", 1))


def p1340(arr: list[int], d: int) -> int:
    """
    1340. Jump Game V https://leetcode.com/problems/jump-game-v/

    Lessons learned:
    - Going to try a simple BFS first, just to build problem understanding.
    - The hint suggests using DP, but I don't see it yet. The problem structure is

          dp[i] = 1 + max(dp[j] for j in range(i - d, i + d + 1) if arr[j] < arr[i])

    TODO

    Examples:
    >>> p1340([6,4,14,6,8,13,9,7,10,6,12], 2)
    4
    >>> p1340([3,3,3,3,3], 3)
    1
    >>> p1340([7,6,5,4,3,2,1], 1)
    7
    """
    if len(arr) == 1:
        return 1

    value_ix = defaultdict(list)
    for ix, val in enumerate(arr):
        value_ix[val].append(ix)

    values = sorted(value_ix.keys())


def p1345(arr: list[int]) -> int:
    """
    1345. Jump Game IV https://leetcode.com/problems/jump-game-iv/

    Lessons learned:
    - I did this with a priority queue, but it's not necessary. A BFS would work
    just as well.
    - You can also do a bidirectional BFS, which can be faster. This means
    building a frontier of nodes from both the start and the end.

    Examples:
    >>> p1345([100,-23,-23,404,100,23,23,23,3,404])
    3
    >>> p1345([7])
    0
    >>> p1345([7,6,9,6,9,6,9,7])
    1
    >>> p1345([7,7,2,1,7,7,7,3,4,1])
    3
    """
    if len(arr) == 1:
        return 0

    value_ix = defaultdict(list)
    for ix, val in enumerate(arr):
        value_ix[val].append(ix)

    seen = set()
    queue = PriorityQueue()
    queue.put((0, 0))
    while queue:
        jumps, ix = queue.get()

        if ix == len(arr) - 1:
            return jumps

        seen.add(ix)

        for ix_ in [ix + 1, ix - 1] + value_ix[arr[ix]]:
            if 0 <= ix_ < len(arr) and ix_ not in seen:
                queue.put((jumps + 1, ix_))

        del value_ix[arr[ix]]

    return -1


def p1456(s: str, k: int) -> int:
    """
    1456. Maximum Number of Vowels in a Substring of Given Length https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/

    Lessons learned:
    - Sliding window and no need for a queue here, because sum statistics are easy
    to update.

    Examples:
    >>> p1456("abciiidef", 3)
    3
    >>> p1456("aeiou", 2)
    2
    >>> p1456("leetcode", 3)
    2
    >>> p1456("rhythms", 4)
    0
    >>> p1456("tryhard", 4)
    1
    """
    vowels = set("aeiou")
    num_vowels = sum(c in vowels for c in s[:k])
    max_vowels = num_vowels
    for i in range(k, len(s)):
        if s[i - k] in vowels:
            num_vowels -= 1
        if s[i] in vowels:
            num_vowels += 1
        max_vowels = max(max_vowels, num_vowels)
    return max_vowels


def p1491(salary: list[int]) -> float:
    """
    1491. Average Salary Excluding the Minimum and Maximum Salary https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/

    Lessons learned:
    - Slightly surprised the single pass Python-loop approach is slightly faster
    than the three pass approach using built-ins.

    Examples:
    >>> p1491([4000,3000,1000,2000])
    2500.0
    >>> p1491([1000,2000,3000])
    2000.0
    """
    return (sum(salary) - min(salary) - max(salary)) / (len(salary) - 2)


def p1491_2(salary: list[int]) -> float:
    """
    Examples:
    >>> p1491_2([4000,3000,1000,2000])
    2500.0
    >>> p1491_2([1000,2000,3000])
    2000.0
    """
    lo, hi = float("inf"), float("-inf")
    sums = 0
    count = 0
    for s in salary:
        if s < lo:
            lo = s
        if s > hi:
            hi = s
        sums += s
        count += 1

    return (sums - lo - hi) / (count - 2)


def p1498(nums: list[int], target: int) -> int:
    """
    1498. Number of Subsequences That Satisfy the Given Sum Condition https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/

    Lessons learned:
    - I had the rough idea, but I was tired, so I looked at a hint.
    - 1 << n is much faster in Python than 2**n.

    Examples:
    >>> p1498([3,5,6,7], 9)
    4
    >>> p1498([3,3,6,8], 10)
    6
    >>> p1498([2,3,3,4,6,7], 12)
    61
    >>> p1498([14,4,6,6,20,8,5,6,8,12,6,10,14,9,17,16,9,7,14,11,14,15,13,11,10,18,13,17,17,14,17,7,9,5,10,13,8,5,18,20,7,5,5,15,19,14], 22)
    272187084
    """
    nums.sort()
    lo, hi = 0, len(nums) - 1
    count = 0
    while lo <= hi:
        if nums[lo] + nums[hi] <= target:
            count += 1 << (hi - lo)
            lo += 1
        else:
            hi -= 1
    return count % (10**9 + 7)


def p1544(s: str) -> str:
    """
    1544. Make The String Great https://leetcode.com/problems/make-the-string-great/

    Examples:
    >>> p1544("leEeetcode")
    'leetcode'
    >>> p1544("abBAcC")
    ''
    >>> p1544("s")
    's'
    """
    stack = []
    for c in s:
        if stack and stack[-1].lower() == c.lower() and stack[-1] != c:
            stack.pop()
        else:
            stack.append(c)
    return "".join(stack)


def p1557(n: int, edges: list[list[int]]) -> list[int]:
    """
    1557. Minimum Number of Vertices to Reach All Nodes https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/

    Lessons learned:
    - At first I thought this required union find, but that is for partitions /
    undirected graphs. After fiddling with a modification of union find for a
    while, I saw that the solution was actually really simple.

    Examples:
    >>> p1557(6, [[0,1],[0,2],[2,5],[3,4],[4,2]])
    [0, 3]
    >>> p1557(5, [[0,1],[2,1],[3,1],[1,4],[2,4]])
    [0, 2, 3]
    """
    nodes_with_parents = set()
    for _, v in edges:
        nodes_with_parents.add(v)

    return [i for i in range(n) if i not in nodes_with_parents]


def p1572(mat: list[list[int]]) -> int:
    """
    1572. Matrix Diagonal Sum https://leetcode.com/problems/matrix-diagonal-sum/

    Examples:
    >>> p1572([[1,2,3],[4,5,6],[7,8,9]])
    25
    >>> p1572([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    8
    """
    n = len(mat)

    if n == 1:
        return mat[0][0]

    total = 0

    for i in range(n):
        total += mat[i][i] + mat[n - 1 - i][i]

    if n % 2 == 1:
        total -= mat[n // 2][n // 2]

    return total


def p1579(n: int, edges: list[list[int]]) -> int:
    """
    1579. Remove Max Number of Edges to Keep Graph Fully Traversable https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/

    Lessons learned:
    - We can build a spanning tree greedily by adding edges when they don't create
    a cycle. We can detect when an edge would create a cycle, by using a
    disjoint set. Counting these edges gives us the number removable edges. This
    problem adds a minor complication by having three types of edges. This
    complication can be dealth with by keeping track of two graphs. Since
    sometimes one edge of type 3 can make two edges of type 1 and 2 obsolete, we
    prioritize adding edges of type 3 first.
    - A spanning tree always has the minimum number of edges to connect all nodes,
    which is V - 1 for a graph with V nodes

    Examples:
    >>> p1579(4, [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]])
    2
    >>> p1579(4, [[3,1,2],[3,2,3],[1,1,4],[2,1,4]])
    0
    >>> p1579(4, [[3,2,3],[1,1,2],[2,3,4]])
    -1
    >>> p1579(2, [[1,1,2],[2,1,2],[3,1,2]])
    2
    """

    def find(x: int, parent: list[int]) -> int:
        while True:
            if parent[x] == x:
                return x
            parent[x] = parent[parent[x]]
            x = parent[x]

    def union(x: int, y: int, parent: list[int]) -> bool:
        """Return True if new connection made."""
        x_root, y_root = find(x, parent), find(y, parent)
        if x_root == y_root:
            return False
        parent[x_root] = y_root
        return True

    alice_graph = list(range(n))
    bob_graph = list(range(n))
    total_edges = 0
    for edge_type, s, t in edges:
        if edge_type == 3:
            ag = union(s - 1, t - 1, alice_graph)
            bg = union(s - 1, t - 1, bob_graph)
            if not (ag or bg):
                total_edges += 1

    for edge_type, s, t in edges:
        if edge_type == 1:
            if not union(s - 1, t - 1, alice_graph):
                total_edges += 1
        elif edge_type == 2:
            if not union(s - 1, t - 1, bob_graph):
                total_edges += 1
        else:
            continue

    def count(parent: list[int]) -> int:
        return len({find(i, parent) for i in range(n)})

    if count(alice_graph) > 1 or count(bob_graph) > 1:
        return -1

    return total_edges


def p1680(n: int) -> int:
    """
    1680. Concatenation of Consecutive Binary Numbers https://leetcode.com/problems/concatenation-of-consecutive-binary-numbers/

    Examples:
    >>> p1680(1)
    1
    >>> p1680(3)
    27
    >>> p1680(12)
    505379714
    """
    M = 10**9 + 7
    total = 1
    for i in range(2, n + 1):
        total = ((total << math.floor(math.log2(i)) + 1) + i) % M

    return total


def p1697(n: int, edgeList: list[list[int]], queries: list[list[int]]) -> list[bool]:
    """
    1697. Checking Existence of Edge Length Limited Paths https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/

    Lessons learned:
    - This problem is a connected component problem, though the weighted edges may
    throw you off. Since we're not looking for total path distance, for each
    query in order of increasing threshold, we can build a graph and calculate
    the connected components given by the query threshold. This lets us build on
    the work done for previous queries.

    Examples:
    >>> p1697(3, [[0,1,2],[1,2,4],[2,0,8],[1,0,16]], [[0,1,2],[0,2,5]])
    [False, True]
    >>> p1697(5, [[0,1,10],[1,2,5],[2,3,9],[3,4,13]], [[0,4,14],[1,4,13]])
    [True, False]
    >>> p1697(3, [[0,1,2],[1,2,4],[2,0,8],[1,0,16]], [[0,2,1], [0,2,7]])
    [False, True]
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while True:
            if x == parent[x]:
                return x
            parent[x] = parent[parent[x]]
            x = parent[x]

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    edgeList.sort(key=lambda x: x[2])
    queries = sorted((q[2], q[0], q[1], i) for i, q in enumerate(queries))

    result = [False] * len(queries)
    i = 0
    for d, q0, q1, j in queries:
        while i < len(edgeList) and edgeList[i][2] < d:
            union(edgeList[i][0], edgeList[i][1])
            i += 1
        result[j] = find(q0) == find(q1)

    return result


def get_moves_list(nums: list[int], k: int) -> int:
    """Test three ways to calculate the absolute value distance.

    For testing 1703.

    Examples:
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 2)
    >>> a == b == c
    True
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 3)
    >>> a == b == c
    True
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 4)
    >>> a == b == c
    True
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 5)
    >>> a == b == c
    True
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 6)
    >>> a == b == c
    True
    >>> a, b, c = get_moves_list([0,0,0,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1], 8)
    >>> a == b == c
    True
    """

    def calc1(stack: deque, k: int) -> int:
        """Calculate using the absolute value formula with the indices."""
        mid = k // 2
        return sum(abs(stack[mid] - i) for i in stack)

    def method1(nums: list[int], k: int) -> list[int]:
        """Calculate using the absolute value formula with the indices."""
        if len(nums) == k or k == 1:
            return 0

        moves = []
        stack = deque()
        s = 0
        for i, n in enumerate(nums):
            if n == 1:
                stack.append(i)
                if stack and len(stack) > k:
                    s = s - stack.popleft()
                if len(stack) == k:
                    moves.append(calc1(stack, k))

        return moves

    def calc2(gaps: list, k: int, j: int) -> int:
        """Calculate using the absolute value formula with the differences between indices."""
        return sum(min(i + 1, k - (i + 1)) * gaps[j + i] for i in range(k - 1))

    def method2(nums: list[int], k: int) -> list[int]:
        """Calculate using the absolute value formula with the differences between indices."""
        gaps = get_gaps(nums)
        moves = []
        for j in range(len(gaps) - k + 2):
            moves.append(calc2(gaps, k, j))
        return moves

    def method3(nums: list[int], k: int) -> list[int]:
        """Calculate using a magic iterative approach."""
        gaps = get_gaps(nums)

        if len(gaps) == k - 1:
            return [calc2(gaps, k, 0)]

        moves = [calc2(gaps, k, 0), calc2(gaps, k, 1)]
        delta1 = moves[-1] - moves[-2]
        for j in range(0, len(gaps) - k):
            if k % 2 == 1:
                delta2 = gaps[j] - gaps[j + k // 2] - gaps[j + k // 2 + 1] + gaps[j + k]
            else:
                delta2 = gaps[j] - 2 * gaps[j + k // 2] + gaps[j + k]
            delta1 = delta1 + delta2
            moves.append(moves[-1] + delta1)
        return moves

    return method1(nums, k), method2(nums, k), method3(nums, k)


def get_gaps(nums: list[int]) -> list[int]:
    gaps = []
    prev = -1
    for i, x in enumerate(nums):
        if prev == -1 and x == 1:
            prev = i
        elif prev != -1 and x == 1:
            gaps.append(i - prev)
            prev = i
    return gaps


def p1703(nums: list[int], k: int) -> int:
    """
    1703. Minimum Adjacent Swaps for K Consecutive Ones https://leetcode.com/problems/minimum-adjacent-swaps-for-k-consecutive-ones/

    Lesson learned:
    - The first solution is not fast enough, but has the right general idea. It
      uses a sliding window of size k that records the indices of the 1s and finds
      the the absolute value of the distances to the median index in that window.
    - The second solution makes two changes. The first is to change focus from
      storing the indices of the 1s to storing the gaps between the indices of the
      1s. This is because the absolute value distance to the median has a formula
      in terms of the differences between the indices. For example, with nums =
      [1, 1, 0, 0, 1, 0, 1, 0, 1], we have gaps = [1, 3, 2, 2], so for k = 5, the
      distance to the median (the third 1) is 1 + 2 * 3 + 2 * 2 + 2 = 13. In
      general, we have

          av(j)   = sum_{i=0}^{k-2} min(i + 1, k - (i + 1)) * gaps[j + i]

      Furthermore, d1(j) = av(j) - av(j-1) and d2(j) = d1(j) - d1(j-1) can be
      expressed as

          d1(j)   = - sum_{i=0}^{k//2 - 1} gaps[j + i - 1] +                          if k is odd
                      sum_{i=k//2}^{k-1} gaps[j + i - 1],
                  = - sum_{i=0}^{k//2 - 1} gaps[j + i - 1] +                          else
                      sum_{i=k//2 + 1}^{k-1} gaps[j + i - 1],
          d2(j)   = gaps[j] - gaps[j + k//2] - gaps[j + k//2 + 1] + gaps[j + k],      if k is odd
                  = gaps[j] - 2 * gaps[j + k//2] + gaps[j + k],                       else

      This means that if we calculate the first two absolute value distances, we
      can calculate the subsequent ones via the formula

          av(j)   = av(j-1) + d1(j-1) + d2(j-1),          j >= 2
          d1(1)   = av(1) - av(0).

      This update step requires us to sum only up to 4 elements from the gaps
      array, as opposed to k.
    - The absolute value distance needs a correction to equal the swap distance

          av      = swap - T(L) - T(R), where T(x) = x * (x + 1) // 2

      where L, R are number of neighbors on the left and right of the median,
      respectively. For example, if the k-stack is [1, 1, 0, 0, 1, 0, 1], the
      absolute value distance to the median is 9, while the actual swaps needed is
      5. The correction factor can be found T(L) + T(R) = T(2) + T(1) = 3 + 1 = 4.
    - I solved this problem on my own, so I feel pretty proud of it.
    - Taking a look at other people's solutions, I see simpler approaches. For
      example, it turns out that the absolute value distance to the median can be
      expressed as

          av(j)   = sum(pos[j+k//2:j+k]) - sum(pos[j:j+k//2])
          pos(j)  = position of the jth 1 in nums

      from which it would be much easier to calculate the iterative approach. The
      way to arrive at this, is to notice that the median point has half the 1s to
      its left and half to its right.
    - There is a competitive programming trick to convert a problem from "min sum
      of moves to make numbers consecutive" to "max sum of moves to a single
      point" and that is to transform pos(j) to pos(j) - j. This is because

          sum pos(j) - j = sum pos(j) - sum j = sum pos(j) - k * (k + 1) // 2

      which is the correction factor we discused above.

    Examples:
    >>> p1703([1,0,0,1,0,1], 2)
    1
    >>> p1703([1,0,0,0,0,0,1,1], 3)
    5
    >>> p1703([1,1,0,1], 2)
    0
    >>> p1703([0,0,0,1,0,1,1,0,1], 3)
    1
    >>> p1703([0,0,0,1,0,1,1,0,1], 4)
    2
    >>> p1703([1,0,1,0,1,0,0,0,1], 4)
    6
    """
    if len(nums) == k or k == 1:
        return 0

    def triangular_number(n: int) -> int:
        return n * (n + 1) // 2

    mid1, mid2 = k // 2 - 1, k // 2
    if k % 2 == 0:
        correction_factor = triangular_number(mid1) + triangular_number(mid2)
    else:
        correction_factor = 2 * triangular_number(mid2)

    def calculate_swaps(stack: deque[int]) -> int:
        """Calculate the swaps needed to get to the median(s) of the stack.

        The stack represents the indices of the 1s in the array.
        """
        return sum(abs(stack[mid2] - i) for i in stack)

    stack = deque()
    s = 0
    m = float("inf")
    for i, n in enumerate(nums):
        if n == 1:
            stack.append(i)
            if stack and len(stack) > k:
                s = s - stack.popleft()
            if len(stack) == k:
                m = min(m, calculate_swaps(stack))

    return m - correction_factor


def p1703_2(nums: list[int], k: int) -> int:
    """
    Examples:
    >>> p1703_2([1,0,0,1,0,0], 2)
    2
    >>> p1703_2([1,0,0,1,0,1], 2)
    1
    >>> p1703_2([1,0,0,0,0,0,1,1], 3)
    5
    >>> p1703_2([1,1,0,1], 2)
    0
    >>> p1703_2([0,0,0,1,0,1,1,0,1], 3)
    1
    >>> p1703_2([0,0,0,1,0,1,1,0,1], 4)
    2
    >>> p1703_2([1,0,1,0,1,0,0,0,1], 4)
    6
    """
    if len(nums) == k or k == 1:
        return 0

    def triangular_number(n: int) -> int:
        return n * (n + 1) // 2

    mid1, mid2 = k // 2 - 1, k // 2
    if k % 2 == 0:
        correction_factor = triangular_number(mid1) + triangular_number(mid2)
    else:
        correction_factor = 2 * triangular_number(mid2)

    def calc2(gaps: list, k: int, j: int) -> int:
        """Calculate using the absolute value formula with the differences between indices."""
        return sum(min(i + 1, k - (i + 1)) * gaps[j + i] for i in range(k - 1))

    gaps = get_gaps(nums)

    if len(gaps) == k - 1:
        return calc2(gaps, k, 0) - correction_factor

    m0, m1 = calc2(gaps, k, 0), calc2(gaps, k, 1)
    delta1 = m1 - m0
    m = min(m0, m1)

    for j in range(0, len(gaps) - k):
        if k % 2 == 1:
            delta2 = gaps[j] - gaps[j + k // 2] - gaps[j + k // 2 + 1] + gaps[j + k]
        else:
            delta2 = gaps[j] - 2 * gaps[j + k // 2] + gaps[j + k]
        delta1 = delta1 + delta2
        m1 = m1 + delta1
        m = min(m, m1)

    return m - correction_factor


def p1706(grid: list[list[int]]) -> list[int]:
    """
    1706. Where Will The Ball Fall https://leetcode.com/problems/where-will-the-ball-fall/

    Examples:
    >>> p1706([[-1]])
    [-1]
    >>> p1706([[1,1,1,-1,-1],[1,1,1,-1,-1],[-1,-1,-1,1,1],[1,1,1,1,-1],[-1,-1,-1,-1,-1]])
    [1, -1, -1, -1, -1]
    >>> p1706([[1,1,1,1,1,1]])
    [1, 2, 3, 4, 5, -1]
    >>> p1706([[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1]])
    [0, 1, 2, 3, 4, -1]
    """

    def find_ball(i: int) -> Literal[-1, 1]:
        x, y = 0, i
        while True:
            if grid[x][y] == 1:
                if y == len(grid[0]) - 1 or grid[x][y + 1] == -1:
                    return -1
                y += 1
            else:
                if y == 0 or grid[x][y - 1] == 1:
                    return -1
                y -= 1
            x += 1
            if x == len(grid):
                return y

    return [find_ball(i) for i in range(len(grid[0]))]


def p1721(head: ListNode, k: int) -> ListNode:
    """
    1721. Swapping Nodes in a Linked List https://leetcode.com/problems/swapping-nodes-in-a-linked-list/

    Lessons learned:
    - Two pointers allows you to do this in one pass.

    Examples:
    >>> listnode_to_list(p1721(ListNode.from_list([1,2,3,4,5]), 2))
    [1, 4, 3, 2, 5]
    >>> listnode_to_list(p1721(ListNode.from_list([7,9,6,6,7,8,3,0,9,5]), 5))
    [7, 9, 6, 6, 8, 7, 3, 0, 9, 5]
    """
    p1 = head
    for _ in range(k - 1):
        p1 = p1.next
    node1 = p1
    p2 = head
    while p1.next:
        p1 = p1.next
        p2 = p2.next
    node2 = p2
    node1.val, node2.val = node2.val, node1.val
    return head


def p1822(nums: list[int]) -> int:
    """
    1822. Sign of the Product of an Array https://leetcode.com/problems/sign-of-the-product-of-an-array/

    Examples:
    >>> p1822([-1,-2,-3,-4,3,2,1])
    1
    >>> p1822([1,5,0,2,-3])
    0
    >>> p1822([-1,1,-1,1,-1])
    -1
    """
    pos = 1
    for n in nums:
        if n < 0:
            pos *= -1
        elif n == 0:
            return 0
    return pos


def p2130(head: ListNode | None) -> int:
    """
    2130. Maximum Twin Sum of a Linked List https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/

    Lessons learned:
    - Finding the midpoint of a linked list can be done with two pointers.
    Reversing a linked list is pretty easy. These steps above can be done in one
    pass.

    Examples:
    >>> p2130(ListNode.from_list([5,4,2,1]))
    6
    >>> p2130(ListNode.from_list([4,2,2,3]))
    7
    >>> p2130(ListNode.from_list([1,100000]))
    100001
    """
    if head is None:
        return 0

    # Find the midpoint
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse the second half
    prev = None
    while slow:
        slow.next, prev, slow = prev, slow, slow.next

    # Find the maximum sum
    m = 0
    while prev:
        m = max(m, prev.val + head.val)
        prev = prev.next
        head = head.next

    return m


def p2131(words: list[str]) -> int:
    """
    2131. Longest Palindrome by Concatenating Two Letter Words https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/

    Examples:
    >>> p2131(["ab","ba","aa","bb","cc"])
    6
    >>> p2131(["ab","ba","cc","ab","ba","cc"])
    12
    >>> p2131(["aa","ba"])
    2
    >>> p2131(["ba", "ce"])
    0
    >>> p2131(["lc","cl","gg"])
    6
    >>> p2131(["ab","ty","yt","lc","cl","ab"])
    8
    >>> p2131(["cc","ll","xx"])
    2
    """
    d: Counter[str] = Counter()

    for word in words:
        d[word] += 1

    total = 0
    extra_double = False
    for key in d:
        if key[0] == key[1]:
            total += d[key] // 2 * 4
            if d[key] % 2 == 1:
                extra_double = True
        elif key == "".join(sorted(key)):
            total += min(d[key], d[key[::-1]]) * 4

    if extra_double:
        total += 2

    return total


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


def p2493(n: int, edges: list[list[int]]) -> int:
    """
    2493. Divide Nodes Into The Maximum Number of Groups https://leetcode.com/problems/divide-nodes-into-the-maximum-number-of-groups/

    Lessons learned:
    - This problem is a pretty straightforward extension of (785 Bipartite Graph
    Checking).
    - I spent a good 30-45 minutes not realizing that I was returning the minimum
    groups instead of maximum. Derp.
    - Checking only nodes with minimum degree <= node degree <= minimum degree + 1
    in a given partition yields a substantial savings (98 percentile
    performance), but I don't quite know if this is a general property or just a
    heuristic that helps on this particular test set. The intuition is that we
    can maximize graph diameter by starting at a leaf, leaves have degree 1, and
    so maybe looking for the smallest degree nodes leads us to outer boundary of
    a graph. Not sure.

    Examples:
    >>> p2493(6, [[1,2],[1,4],[1,5],[2,6],[2,3],[4,6]])
    4
    >>> p2493(3, [[1,2],[2,3],[3,1]])
    -1
    """
    # Convert from edge list to adjacency list
    graph = defaultdict(list)

    for u, v in edges:
        graph[u - 1].append(v - 1)
        graph[v - 1].append(u - 1)

    # Find connected components (store the lowest index member)
    seen: set[int] = set()
    partitions: dict[int, set[int]] = defaultdict(set)

    def dfs_connected_components(node: int, partition: int):
        if node not in seen:
            seen.add(node)
            partitions[partition].add(node)
            for neighbor_node in graph[node]:
                dfs_connected_components(neighbor_node, partition)

    for node in range(n):
        if node not in seen:
            dfs_connected_components(node, node)

    # Get a coloring for each connected component {partition: {node: color}}
    coloring = {}

    def bfs_coloring(root: int) -> bool:
        queue: deque[tuple[int, int]] = deque()
        queue.append((root, 0))
        while queue:
            node, color = queue.popleft()

            if node not in coloring:
                coloring[node] = color

                for neighbor_node in graph[node]:
                    if (
                        neighbor_node in coloring
                        and (coloring[neighbor_node] - color - 1) % 2 == 1
                    ):
                        return False
                    if neighbor_node not in coloring:
                        queue.append((neighbor_node, color + 1))

        return True

    # Do BFS from every node, building a spanning tree, and looking for the maximum depth achieved
    result = 0
    max_coloring = -1
    for _, partition_nodes in partitions.items():
        for node in partition_nodes:
            if not bfs_coloring(node):
                return -1

            max_coloring = max(max_coloring, max(coloring.values()) + 1)
            coloring = {}
        result += max_coloring
        max_coloring = -1

    # A little degree checking heuristic that gives a big boost, but might not work in general.
    # result = 0
    # max_coloring = -1
    # for _, partition_nodes in partitions.items():
    #     min_degree = min(len(graph[node]) for node in partition_nodes)
    #     check_nodes = [node for node in partition_nodes if len(graph[node]) <= min_degree + 1]

    #     for node in check_nodes:
    #         if not bfs_coloring(node):
    #             return -1
    #         else:
    #             max_coloring = max(max_coloring, max(coloring.values()) + 1)
    #             coloring = defaultdict(dict)
    #     result += max_coloring
    #     max_coloring = -1

    return result


def p2540(nums1: list[int], nums2: list[int]) -> int:
    """
    2540. Minimum Common Value https://leetcode.com/problems/minimum-common-value/

    Examples:
    >>> p2540([1, 2, 3], [2, 4])
    2
    >>> p2540([1, 2, 3], [4, 5])
    -1
    >>> p2540([1, 2, 3, 6], [2, 3, 4, 5])
    2
    >>> p2540([1, 1, 2], [2, 4])
    2
    """
    n, m = len(nums1), len(nums2)
    i, j = 0, 0
    while i < n and j < m:
        if nums1[i] == nums2[j]:
            return nums1[i]
        if nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1

    return -1


def p2608(n: int, edges: list[list[int]]) -> int:
    """
    2608. Shortest Cycle in a Graph https://leetcode.com/problems/shortest-cycle-in-a-graph/

    TODO

    Examples:
    >>> p2608(7, [[0,1],[1,2],[2,0],[3,4],[4,5],[5,6],[6,3]])
    3
    >>> p2608(4, [[0,1],[0,2]])
    -1
    """
    ...


def p3005(nums: list[int]) -> int:
    """
    3005. Count Elements With Maximum Frequency https://leetcode.com/problems/count-elements-with-maximum-frequency

    Examples:
    >>> p3005([1, 2, 2, 3, 1, 4])
    4
    >>> p3005([1, 2, 3, 4, 5])
    5
    """
    c = Counter(nums)
    max_value = max(c.values())
    return sum(v for v in c.values() if v == max_value)
