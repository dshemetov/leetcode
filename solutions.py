# %%
import math
from array import array
from bisect import bisect_left, insort
from collections.abc import Generator, Iterable
from collections import Counter, defaultdict, deque, namedtuple
from fractions import Fraction
from functools import reduce
from itertools import permutations
from operator import mul
from typing import Iterable, Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# %% 1. Two Sum https://leetcode.com/problems/two-sum/
def two_sum(nums: list[int], target: int) -> list[int]:
    """1. Two Sum https://leetcode.com/problems/two-sum/

    Examples:
    >>> two_sum([3, 3], 6)
    [0, 1]
    >>> two_sum([3, 2, 4], 7)
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


# %% 2. Add Two Numbers https://leetcode.com/problems/add-two-numbers/
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def add_two_numbers(l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
    """2. Add Two Numbers https://leetcode.com/problems/add-two-numbers/

    Examples:
    >>> list_to_int(add_two_numbers(int_to_list(0), int_to_list(15)))
    15
    >>> list_to_int(add_two_numbers(int_to_list(12), int_to_list(15)))
    27
    >>> list_to_int(add_two_numbers(int_to_list(12), int_to_list(153)))
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


def int_to_list(i: int) -> ListNode:
    num_list = None
    for x in str(i):
        num_list = ListNode(val=int(x), next=num_list)
    return num_list


def list_to_int(l: ListNode) -> int:
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
    while l:
        num += l.val * 10**digit
        digit += 1
        l = l.next
    return num


# %% 3. Longest Substring Without Repeating Characters https://leetcode.com/problems/longest-substring-without-repeating-characters/
def length_of_longest_substring(s: str) -> int:
    """3. Longest Substring Without Repeating Characters https://leetcode.com/problems/longest-substring-without-repeating-characters/

    Examples:
    >>> length_of_longest_substring("a")
    1
    >>> length_of_longest_substring("aa")
    1
    >>> length_of_longest_substring("aaa")
    1
    >>> length_of_longest_substring("aab")
    2
    >>> length_of_longest_substring("abba")
    2
    >>> length_of_longest_substring("abccba")
    3
    >>> length_of_longest_substring("au")
    2
    >>> length_of_longest_substring("cdd")
    2
    >>> length_of_longest_substring("abcabcbb")
    3
    >>> length_of_longest_substring("aabcdef")
    6
    >>> length_of_longest_substring("abcdeffff")
    6
    >>> length_of_longest_substring("dvdf")
    3
    >>> length_of_longest_substring("ohomm")
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


# %% 4. Median of Two Sorted Arrays https://leetcode.com/problems/median-of-two-sorted-arrays/
def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    """4. Median of Two Sorted Arrays https://leetcode.com/problems/median-of-two-sorted-arrays/

    Lessons learned:
    - Swapping two variables in Python is fast and swaps pointers under the hood using an auxiliary variable: https://stackoverflow.com/a/62038590/4784655.
    - I had a decent idea at first, but needed a hint about partition finding to really figure it out.
    - Thanks to NeetCode https://www.youtube.com/watch?v=q6IEA26hvXc for the explanation.

    The key is to do a binary search on joint partitions of the arrays.
    A partition on nums1 is given by all elements nums[:mid1] and nums[mid1:].
    Similarly on nums2 with mid2.
    The median is the partition that satisfies:
    - len(nums1[:mid1]) + len(nums2[:mid2]) == len(nums1[mid1:]) + len(nums2[mid2:])
    - nums1[mid1 - 1] <= nums2[mid2]
    - nums2[mid2 - 1] <= nums1[mid1]

    Examples:
    >>> find_median_sorted_arrays([1, 3], [2])
    2.0
    >>> find_median_sorted_arrays([1, 2], [3, 4])
    2.5
    >>> find_median_sorted_arrays([1, 3], [2, 4])
    2.5
    >>> a1 = [5, 13, 15]
    >>> b1 = [0, 10, 10, 15, 20, 20, 25]
    >>> find_median_sorted_arrays(a1, b1) == get_median_sorted(sorted(a1 + b1))
    True
    >>> a2 = [9, 36, 44, 45, 51, 67, 68, 69]
    >>> b2 = [7, 20, 26, 27, 30, 43, 54, 73, 76, 88, 91, 94]
    >>> find_median_sorted_arrays(a2, b2) == get_median_sorted(sorted(a2 + b2))
    True
    >>> a2 = [2, 2, 2, 2, 2, 2, 5]
    >>> b2 = [0, 0, 0, 0, 0, 0, 0, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    >>> find_median_sorted_arrays(a2, b2) == get_median_sorted(sorted(a2 + b2))
    True
    >>> a2 = [2, 2, 2, 4, 5, 7, 8, 9]
    >>> b2 = [1, 1, 1, 1, 1, 3, 6, 10, 11, 11, 11, 11]
    >>> find_median_sorted_arrays(a2, b2) == get_median_sorted(sorted(a2 + b2))
    True
    """
    if not nums1:
        return get_median_sorted(nums2)
    if not nums2:
        return get_median_sorted(nums1)

    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    lo, hi = 0, len(nums1)
    while lo <= hi:
        mid1 = (lo + hi) // 2
        # This is not len(nums2) - mid1 because we are splitting the total number of elements in half.
        mid2 = (len(nums1) + len(nums2) + 1) // 2 - mid1

        a_left = nums1[mid1 - 1] if mid1 > 0 else float("-inf")
        a_right = nums1[mid1] if mid1 < len(nums1) else float("inf")
        b_left = nums2[mid2 - 1] if mid2 > 0 else float("-inf")
        b_right = nums2[mid2] if mid2 < len(nums2) else float("inf")

        if a_left <= b_right and b_left <= a_right:
            if (len(nums1) + len(nums2)) % 2 == 0:
                return (max(a_left, b_left) + min(a_right, b_right)) / 2
            else:
                return float(max(a_left, b_left))
        elif a_left > b_right:
            hi = mid1 - 1
        else:
            lo = mid1 + 1


def get_median_sorted(nums: list[int]) -> float:
    if len(nums) == 1:
        return nums[0]

    mid = len(nums) // 2
    if len(nums) % 2 == 0:
        return (nums[mid] + nums[mid - 1]) / 2
    else:
        return nums[mid]


# %% 8. String to Integer (atoi) https://leetcode.com/problems/string-to-integer-atoi/
def my_atoi(s: str) -> int:
    """8. String to Integer (atoi) https://leetcode.com/problems/string-to-integer-atoi/

    Examples:
    >>> my_atoi("42")
    42
    >>> my_atoi("   -42")
    -42
    >>> my_atoi("4193 with words")
    4193
    >>> my_atoi("words and 987")
    0
    >>> my_atoi("-91283472332")
    -2147483648
    >>> my_atoi("91283472332")
    2147483647
    >>> my_atoi("3.14159")
    3
    >>> my_atoi("+-2")
    0
    >>> my_atoi("  -0012a42")
    -12
    >>> my_atoi("  +0 123")
    0
    >>> my_atoi("-0")
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
            elif c == "-":
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
                elif factor == -1 and num > _min:
                    return -_min
            else:
                break

    return factor * num


# %% 9. Palindrome Number https://leetcode.com/problems/palindrome-number/
def is_palindrome(x: int) -> bool:
    """9. Palindrome Number https://leetcode.com/problems/palindrome-number/

    Examples:
    >>> is_palindrome(121)
    True
    >>> is_palindrome(-121)
    False
    >>> is_palindrome(10)
    False
    >>> is_palindrome(0)
    True
    >>> is_palindrome(1)
    True
    >>> is_palindrome(1331)
    True
    >>> is_palindrome(1332)
    False
    >>> is_palindrome(133454331)
    True
    >>> is_palindrome(1122)
    False
    """
    if x < 0:
        return False
    elif x == 0:
        return True

    num_digits = math.floor(math.log10(x)) + 1

    if num_digits == 1:
        return True
    else:
        for i in range(num_digits // 2):
            if x // 10**i % 10 != x // 10 ** (num_digits - i - 1) % 10:
                return False
        return True


# %% 13. Roman to Integer https://leetcode.com/problems/roman-to-integer/
def roman_to_int(s: str) -> int:
    """13. Roman to Integer https://leetcode.com/problems/roman-to-integer/

    Examples:
    >>> roman_to_int("III")
    3
    >>> roman_to_int("IV")
    4
    >>> roman_to_int("IX")
    9
    >>> roman_to_int("LVIII")
    58
    >>> roman_to_int("MCMXCIV")
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


# %% 14. Longest Common Prefix https://leetcode.com/problems/longest-common-prefix/
def longest_common_prefix(strs: list[str]) -> str:
    """14. Longest Common Prefix https://leetcode.com/problems/longest-common-prefix/

    Examples:
    >>> longest_common_prefix(["flower","flow","flight"])
    'fl'
    >>> longest_common_prefix(["dog","racecar","car"])
    ''
    >>> longest_common_prefix(["dog","dog","dog","dog"])
    'dog'
    """
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
        if prefix == "":
            break
    return prefix


# %% 19. Remove Nth Node From End of List https://leetcode.com/problems/remove-nth-node-from-end-of-list/
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
    """19.  Remove Nth Node From End of List https://leetcode.com/problems/remove-nth-node-from-end-of-list/

    Examples:
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2, 3, 4, 5]), 1))
    [1, 2, 3, 4]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2, 3, 4, 5]), 2))
    [1, 2, 3, 5]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2, 3, 4, 5]), 3))
    [1, 2, 4, 5]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2, 3, 4, 5]), 4))
    [1, 3, 4, 5]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2, 3, 4, 5]), 5))
    [2, 3, 4, 5]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1]), 1))
    []
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2]), 1))
    [1]
    >>> listnode_to_list(remove_nth_from_end(list_to_listnode([1, 2]), 2))
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


def list_to_listnode(l: list[int]) -> ListNode:
    if not l:
        return None

    original_head = head = ListNode(l[0])
    for x in l[1:]:
        head.next = ListNode(x)
        head = head.next

    return original_head


def listnode_to_list(head: ListNode) -> list[int]:
    """
    Examples:
    >>> listnode_to_list(list_to_listnode([1, 2, 3, 4, 5]))
    [1, 2, 3, 4, 5]
    """
    l = []
    while head:
        l.append(head.val)
        head = head.next

    return l


# %% 26. Remove Duplicates from Sorted Array https://leetcode.com/problems/remove-duplicates-from-sorted-array/
def remove_duplicates(nums: list[int]) -> int:
    """26. Remove Duplicates from Sorted Array https://leetcode.com/problems/remove-duplicates-from-sorted-array/

    Examples:
    >>> remove_duplicates([1, 1, 2])
    2
    >>> remove_duplicates([0,0,1,1,1,2,2,3,3,4])
    5
    """
    k = 0
    for i in range(1, len(nums)):
        if nums[k] != nums[i]:
            k += 1
            nums[k] = nums[i]
    return k + 1


# %% 36. Valid Sudoku https://leetcode.com/problems/valid-sudoku/
def is_valid_sudoku(board: list[list[str]]) -> bool:
    mat = np.char.replace(np.array(board), ".", "0").astype(int)

    for i in range(mat.shape[0]):
        if not (np.bincount(mat[i, :])[1:] <= 1).all():
            return False
        if not (np.bincount(mat[:, i])[1:] <= 1).all():
            return False

    for i in range(3):
        for j in range(3):
            if not (np.bincount(mat[3 * i : 3 * i + 3, 3 * j : 3 * j + 3].flatten())[1:] <= 1).all():
                return False

    return True


board = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"],
]
assert is_valid_sudoku(board) == True

board = [
    ["8", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"],
]
assert is_valid_sudoku(board) == False


# %% 49. https://leetcode.com/problems/group-anagrams/
def groupAnagrams(strs: list[str]) -> list[list[str]]:
    def group_key(s: str) -> Counter:
        return tuple(sorted(s))

    groups = defaultdict(list)
    for s in strs:
        groups[group_key(s)].append(s)

    return list(groups.values())


# %% 91. https://leetcode.com/problems/decode-ways/
valid_codes = {str(x) for x in range(1, 27)}


def get_fibonacci_number(n: int) -> int:
    phi = (1 + 5**0.5) / 2
    return int(phi**n / 5**0.5 + 0.5)


# Each one corresponds to a way to decode a string where your tokens are either length 1 or 2.
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


def numDecodings(s: str) -> int:
    """
    Examples:
    >>> numDecodings("0")
    0
    >>> numDecodings("06")
    0
    >>> numDecodings("1")
    1
    >>> numDecodings("12")
    2
    >>> numDecodings("111")
    3
    >>> numDecodings("35")
    1
    >>> numDecodings("226")
    3
    >>> numDecodings("2020")
    1
    >>> numDecodings("2021")
    2
    >>> numDecodings("2022322")
    6
    """

    def get_substrings(s: str) -> Iterable[str]:
        prev = 0
        for i, c in enumerate(s):
            if c in "34567890":
                yield s[prev : i + 1]
                prev = i + 1
        if s[prev:]:
            yield s[prev:]

    return reduce(mul, (num_decodings_recursive(x) for x in get_substrings(s)))


def num_decodings_recursive(s: str) -> int:
    """
    Examples:
    >>> num_decodings_recursive("0")
    0
    >>> num_decodings_recursive("06")
    0
    >>> num_decodings_recursive("1")
    1
    >>> num_decodings_recursive("12")
    2
    >>> num_decodings_recursive("111")
    3
    >>> num_decodings_recursive("35")
    1
    >>> num_decodings_recursive("226")
    3
    >>> num_decodings_recursive("2020")
    1
    >>> num_decodings_recursive("2021")
    2
    >>> num_decodings_recursive("2022322")
    6
    """
    if s[0] == "0":
        return 0

    if len(s) == 1:
        return 1

    if len(s) == 2:
        if int(s) <= 26 and s[1] != "0":
            return 2
        elif int(s) <= 26 and s[1] == "0":
            return 1
        elif int(s) > 26 and s[1] != "0":
            return 1
        else:
            return 0

    if set(s) <= {"1", "2"}:
        return get_fibonacci_number(len(s) + 1)

    if s[:2] in valid_codes:
        return num_decodings_recursive(s[1:]) + num_decodings_recursive(s[2:])
    else:
        return num_decodings_recursive(s[1:])


# A dynamic programming approach from the discussion section.
# See this diagram: https://en.wikipedia.org/wiki/Composition_(combinatorics)#/media/File:Fibonacci_climbing_stairs.svg
# Really comes to the recursive formula d[n] = d[n-1] + d[n-2].
def numDecodings3(s: str) -> int:
    """
    Examples:
    >>> numDecodings3("0")
    0
    >>> numDecodings3("06")
    0
    >>> numDecodings3("1")
    1
    >>> numDecodings3("12")
    2
    >>> numDecodings3("111")
    3
    >>> numDecodings3("35")
    1
    >>> numDecodings3("226")
    3
    >>> numDecodings3("2020")
    1
    >>> numDecodings3("2021")
    2
    >>> numDecodings3("2022322")
    6
    """
    if not s or s[0] == "0":
        return 0

    d = [0] * (len(s) + 1)

    # base case initialization
    d[0:2] = [1, 1]

    for i in range(2, len(s) + 1):
        # One step jump
        if 0 < int(s[i - 1 : i]):  # (2)
            d[i] = d[i - 1]
        # Two step jump
        if 10 <= int(s[i - 2 : i]) <= 26:  # (3)
            d[i] += d[i - 2]

    return d[-1]


# %% 133. https://leetcode.com/problems/clone-graph/
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, neighbors: list['Node'] | None = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def adjacency_list_to_node_graph(adjacency_list: list[list[int]]) -> 'Node':
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


def node_graph_to_adjacency_list(node: 'Node') -> 'Node':
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
    

def cloneGraph(node: 'Node') -> 'Node':
    """
    Examples:
    >>> cloneGraph(None)
    >>> node_graph_to_adjacency_list(cloneGraph(adjacency_list_to_node_graph([[1, 2], [1, 4], [2, 3], [3, 4]])))
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


# %% 151. https://leetcode.com/problems/reverse-words-in-a-string/
# Lesson learned:
# - I keep trying to use a Python for-loop, but the built-ins are just so much faster.
def reverseWords(s: str) -> str:
    """
    Examples:
    >>> reverseWords("the sky is blue")
    'blue is sky the'
    >>> reverseWords("  hello world!  ")
    'world! hello'
    >>> reverseWords("a good   example")
    'example good a'
    >>> reverseWords("  Bob    Loves  Alice   ")
    'Alice Loves Bob'
    """
    stack = []
    words = []
    for i in range(len(s) - 1, -1, -1):
        if s[i] != " ":
            stack.append(s[i])
        elif s[i] == " " and stack:
            words.append("".join(reversed(stack)))
            stack = []
    if stack:
        words.append("".join(reversed(stack)))
    return " ".join(words)


def reverseWords2(s: str) -> str:
    """
    Examples:
    >>> reverseWords("the sky is blue")
    'blue is sky the'
    >>> reverseWords("  hello world!  ")
    'world! hello'
    >>> reverseWords("a good   example")
    'example good a'
    >>> reverseWords("  Bob    Loves  Alice   ")
    'Alice Loves Bob'
    """
    return " ".join(s.split()[::-1])


def reverseWords3(s: str) -> str:
    """
    Follow up: use O(1) extra space.
    - Reverse string.
    - Then reverse each worse.

    Examples:
    >>> reverseWords("the sky is blue")
    'blue is sky the'
    >>> reverseWords("  hello world!  ")
    'world! hello'
    >>> reverseWords("a good   example")
    'example good a'
    >>> reverseWords("  Bob    Loves  Alice   ")
    'Alice Loves Bob'
    """

    a = array("u", [])
    a.fromunicode(s.strip())
    a.reverse()

    def reverse_word(arr: array, lo: int, hi: int) -> None:
        l, h = lo, hi
        while l < h:
            arr[l], arr[h] = arr[h], arr[l]
            l += 1
            h -= 1

    lo = 0
    for i in range(len(a)):
        if a[i] == " ":
            reverse_word(a, lo, i - 1)
            lo = i + 1

    reverse_word(a, lo, len(a) - 1)

    lo, space = 0, 0
    for i in range(len(a)):
        space = space + 1 if a[i] == " " else 0
        if space <= 1:
            a[lo] = a[i]
            lo += 1

    return "".join(a[:lo])


# %% 200. https://leetcode.com/problems/number-of-islands/
def numIslands(grid: list[list[str]]) -> int:
    return len(get_connected_components(grid))


def get_connected_components(matrix: list[list[str]]) -> set[tuple[tuple[int, int], ...]]:
    islands = set()
    explored = set()
    possible_locations = ((i, j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j] == "1")
    for ix in possible_locations:
        if ix not in explored:
            island = get_connected_component_at_index(ix)
            islands |= {tuple(island)}
            explored |= island
    return islands


def get_connected_component_at_index(ix: tuple[int, int], matrix: list[list[str]]) -> set[tuple[int, int]]:
    explored_nodes = set()
    unexplored_nodes = set({ix})
    while len(unexplored_nodes - explored_nodes) > 0:
        node = unexplored_nodes.pop()
        unexplored_nodes |= get_node_neighbors(node, matrix) - explored_nodes
        explored_nodes |= {node}
    return explored_nodes


def get_node_neighbors(ix: tuple[int, int], matrix: list[list[str]]) -> set[tuple[int, int]]:
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = set()
    for dirx, diry in directions:
        nix, niy = (ix[0] + dirx, ix[1] + diry)
        if 0 <= nix < len(matrix) and 0 <= niy < len(matrix[0]):
            if matrix[nix][niy] == "1":
                neighbors |= {(nix, niy)}
    return neighbors


# %% 212. https://leetcode.com/problems/word-search-ii/description/
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    """
    Examples:
    >>> set(findWords([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], ["oath","pea","eat","rain"])) == set(["eat", "oath"])
    True
    >>> findWords([["a","b"],["c","d"]], ["abcb"])
    []
    >>> set(findWords([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], ["oath","pea","eat","rain", "oat", "oatht", "naaoetaerkhi", "naaoetaerkhii"])) == set(["eat", "oath", "oat", "naaoetaerkhi"])
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

    def dfs(i: int, j: int, node: dict, path: str, board: list[list[str]], found_words: set[str]) -> None:
        if node.get("#"):
            found_words.add(path)
            trie.remove(path)

        board[i][j] = "$"

        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ni, nj = (i + di, j + dj)
            if 0 <= ni < len(board) and 0 <= nj < len(board[0]) and board[ni][nj] in node and len(path) < 12:
                dfs(ni, nj, node[board[ni][nj]], path + board[ni][nj], board, found_words)

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

    found_words = set()
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] in trie.root:
                dfs(i, j, trie.root[board[i][j]], board[i][j], board, found_words)

    return list(found_words)


# %% 222. https://leetcode.com/problems/count-complete-tree-nodes/
# Lessons learned:
# - "A complete binary tree is a binary tree in which every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible."
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


def countNodes(root: TreeNode | None) -> int:
    """
    Examples:
    >>> countNodes(make_binary_tree([1,2,3,4,5,6]))
    6
    >>> countNodes(make_binary_tree([1,2,3,4,5,6,None]))
    6
    >>> countNodes(make_binary_tree([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]))
    15
    >>> countNodes(make_binary_tree([1,2,3,4,5,6,7,8,9,10,11,12,None,None,None]))
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
        return True if node else False

    lo, hi = 0, 2 ** (height) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        node_in_tree = is_node_in_tree(root, mid)
        if node_in_tree:
            lo = mid + 1
        else:
            hi = mid - 1

    return 2 ** (height) + lo - 1


# %% 223. https://leetcode.com/problems/rectangle-area/
def computeArea(ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
    A1 = (ax2 - ax1) * (ay2 - ay1)
    A2 = (bx2 - bx1) * (by2 - by1)
    I = max(min(ax2, bx2) - max(ax1, bx1), 0) * max(min(ay2, by2) - max(ay1, by1), 0)
    return A1 + A2 - I


# %% 242. https://leetcode.com/problems/valid-anagram/
def isAnagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)


# %% 263. https://leetcode.com/problems/ugly-number/
def isUgly(n: int) -> bool:
    if n < 1:
        return False
    while n % 2 == 0:
        n /= 2
    while n % 3 == 0:
        n /= 3
    while n % 5 == 0:
        n /= 5
    return True if n == 1 else False


# %% 295. https://leetcode.com/problems/find-median-from-data-stream/
class MedianFinder:
    """
    Examples:
    >>> mf = MedianFinder()
    >>> mf.addNum(1)
    >>> mf.addNum(2)
    >>> mf.findMedian()
    1.5
    >>> mf.addNum(3)
    >>> mf.findMedian()
    2.0
    >>> mf = MedianFinder()
    >>> mf.addNum(1)
    >>> mf.addNum(2)
    >>> mf.addNum(3)
    >>> mf.addNum(4)
    >>> mf.addNum(5)
    >>> mf.addNum(6)
    >>> mf.addNum(7)
    >>> mf.findMedian()
    4.0
    >>> mf = MedianFinder()
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
        else:
            return (self.heap[len(self.heap) // 2] + self.heap[len(self.heap) // 2 - 1]) / 2


# %% 316. https://leetcode.com/problems/remove-duplicate-letters/
def removeDuplicateLetters(s: str) -> str:
    """
    Examples:
    >>> removeDuplicateLetters("bcabc")
    'abc'
    >>> removeDuplicateLetters("cbacdcbc")
    'acdb'
    """
    ...


# %% 345. https://leetcode.com/problems/reverse-vowels-of-a-string/
def reverseVowels(s: str) -> str:
    """
    Examples:
    >>> reverseVowels("hello")
    'holle'
    >>> reverseVowels("leetcode")
    'leotcede'
    """
    if len(s) == 1:
        return s

    hi = len(s) - 1
    s_ = []
    for i in range(len(s)):
        if s[i] in "aeiouAEIOU":
            while s[hi] not in "aeiouAEIOU":
                hi -= 1
            s_.append(s[hi])
            hi -= 1
        else:
            s_.append(s[i])

    return "".join(s_)


# %% 374. https://leetcode.com/problems/guess-number-higher-or-lower/
# Lessons learned:
# - bisect_left has a 'key' argument as of 3.10.
__pick__ = 6


def guess(num: int) -> int:
    if num == __pick__:
        return 0
    elif num > __pick__:
        return -1
    else:
        return 1


def guessNumber(n: int) -> int:
    """
    Examples:
    >>> guessNumber(10)
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


def guessNumber2(n: int) -> int:
    """
    Examples:
    >>> guessNumber2(10)
    6
    """

    return bisect_left(range(0, n), 0, lo=0, hi=n, key=lambda x: -guess(x))


# %% 402. https://leetcode.com/problems/remove-k-digits/
# Idea: remove the top n largest elements, while making sure you don't have leading zeros.
def removeKdigits(num: str, k: int) -> str:
    """
    Examples:
    >>> removeKdigits("1432219", 3)
    '1219'
    >>> removeKdigits("10200", 1)
    '200'
    >>> removeKdigits("10", 2)
    '0'
    """
    if len(num) == k:
        return "0"

    queue = deque(sorted(num[:k]))
    # This is borked.
    for c in num:
        if c < queue[0]:
            queue.appendleft(c)
        elif queue[-1] < c:
            queue.append(c)

    for _ in range(k):
        if num[0] == "0" and len(num) > 1:
            num = num[1:]
        elif num[0] == "0":
            return "0"
        else:
            num.replace(queue.pop(), "", 1)

    return num


# %% 433. https://leetcode.com/problems/minimum-genetic-mutation/
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
        return set(mutation for mutation in bank if sum(1 for i in range(len(mutation)) if mutation[i] != gene[i]) == 1)

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


# %% 587. https://leetcode.com/problems/erect-the-fence/
def outerTrees(trees: list[list[int]]) -> list[list[int]]:
    """
    Convex hull: https://en.wikipedia.org/wiki/Convex_hull_algorithms#Algorithms
    Graham scan: https://en.wikipedia.org/wiki/Graham_scan

    Examples:
    >>> outerTrees([[1,1],[2,2],[2,0],[2,4],[3,3],[4,2]])
    [[1, 1], [2, 0], [3, 3], [2, 4], [4, 2]]
    >>> outerTrees([[1,2],[2,2],[4,2]])
    [[1, 2], [2, 2], [4, 2]]
    >>> outerTrees([[1,1],[2,2],[3,3],[2,1],[4,1],[2,3],[1,3]])
    [[1, 1], [2, 1], [4, 1], [3, 3], [1, 3]]
    """

    def crossproduct(v1: tuple[int, int], v2: tuple[int, int]) -> int:
        return v1[0] * v2[1] - v1[1] * v2[0]

    def polar_angle_from_pt(pt1: tuple[int, int], pt2: tuple[int, int]) -> int:
        return crossproduct(pt1, (pt2[0] - pt1[0], pt2[1] - pt1[1]))

    def ccw(pt1: tuple[int, int], pt2: tuple[int, int], pt3: tuple[int, int]) -> float:
        """Counterclockwise."""
        v1 = (pt2[0] - pt1[0], pt2[1] - pt1[1])
        v2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
        return crossproduct(v1, v2)

    def get_polar_angle(pt1: tuple[int, int], pt2: tuple[int, int]) -> Fraction | float:
        if pt1 == pt2:
            raise ValueError("Undefined polar angle for a point to itself.")
        if pt1[0] == pt2[0] and pt1[1] > pt2[1]:
            return -math.inf
        elif pt1[0] == pt2[0] and pt1[1] < pt2[1]:
            return math.inf
        else:
            return Fraction(pt2[1] - pt1[1], pt2[0] - pt1[0])

    # Get bottom left point.
    bottom_left_pt = (math.inf, math.inf)
    for x, y in trees:
        if y < bottom_left_pt[1] or (y == bottom_left_pt[1] and x < bottom_left_pt[0]):
            bottom_left_pt = (x, y)

    # Sort by polar angle.
    trees = sorted(trees, lambda x: polar_angle_from_pt(bottom_left_pt, x))

    # stack = []
    # new_pt = trees[1]
    # while new_pt != start:

    # def find_minimal_slope(pt: tuple[int, int], slope: float) -> tuple[int, int]:
    #     min_slope = slope
    #     for tree in trees:
    #         if pt[0] == tree[0]:

    # between_trees = []


def crossproduct(v1: tuple[int, int], v2: tuple[int, int]) -> int:
    return v1[0] * v2[1] - v1[1] * v2[0]


def polar_angle_from_pt(pt1: tuple[int, int], pt2: tuple[int, int]) -> int:
    return crossproduct(pt1, (pt2[0] - pt1[0], pt2[1] - pt1[1]))


pts = [(math.cos(t), math.sin(t)) for t in np.linspace(0, 2 * math.pi, 20)]
# Get bottom left point.
bottom_left_pt = (math.inf, math.inf)
for x, y in pts:
    if y < bottom_left_pt[1] or (y == bottom_left_pt[1] and x < bottom_left_pt[0]):
        bottom_left_pt = (x, y)

angles = [polar_angle_from_pt(pts[0], pt) for pt in pts]
import matplotlib.pyplot as plt

plt.plot(angles)

# outerTrees([[1,1],[2,2],[3,3]])
# %% 622. https://leetcode.com/problems/design-circular-queue/
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


cmds = ["MyCircularQueue", "enQueue", "enQueue", "enQueue", "enQueue", "Rear", "isFull", "deQueue", "enQueue", "Rear"]
inputs = [[3], [1], [2], [3], [4], [], [], [], [4], []]
run(cmds, inputs)

cmds = ["MyCircularQueue", "enQueue", "enQueue", "deQueue", "enQueue", "deQueue", "enQueue", "deQueue", "enQueue", "deQueue", "Front"]
inputs = [[2], [1], [2], [], [3], [], [3], [], [3], [], []]
run(cmds, inputs)


# %% 658. https://leetcode.com/problems/find-k-closest-elements/
def findClosestElements(arr: list[int], k: int, x: int) -> list[int]:
    """
    Examples:
    >>> findClosestElements([1, 2, 3, 4, 5], 4, 3)
    [1, 2, 3, 4]
    >>> findClosestElements([1, 2, 3, 4, 5], 4, -1)
    [1, 2, 3, 4]
    >>> findClosestElements([1, 2, 3, 4, 5], 4, 4)
    [2, 3, 4, 5]
    >>> findClosestElements([1, 2, 3, 4, 5], 2, 4)
    [3, 4]
    """

    def find_insertion_index(arr: list[int], x: int) -> int:
        lo, hi = 0, len(arr) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == x:
                return mid
            elif arr[mid] < x:
                lo = mid + 1
            else:
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


def findClosestElements2(arr: list[int], k: int, x: int) -> list[int]:
    """
    Faster, binary search on the window itself.

    I had the idea that this problem is like a minimization problem, but I couldn't make the details work.

    Examples:
    >>> findClosestElements2([1, 2, 3, 4, 5], 4, 3)
    [1, 2, 3, 4]
    >>> findClosestElements2([1, 2, 3, 4, 5], 4, -1)
    [1, 2, 3, 4]
    >>> findClosestElements2([1, 2, 3, 4, 5], 4, 4)
    [2, 3, 4, 5]
    >>> findClosestElements2([1, 2, 3, 4, 5], 2, 4)
    [3, 4]
    >>> findClosestElements2([1, 2, 3, 3, 4, 5, 90, 100], 3, 4)
    [3, 3, 4]
    """
    lo, hi = 0, len(arr) - k
    while lo < hi:
        mid = (lo + hi) // 2
        if x - arr[mid] > arr[mid + k] - x:
            lo = mid + 1
        else:
            hi = mid
    return arr[lo : lo + k]


# %% 766. https://leetcode.com/problems/toeplitz-matrix/
def isToeplitzMatrix(matrix: list[list[int]]) -> bool:
    """
    Examples:
    >>> isToeplitzMatrix([[1, 2, 3, 4], [5, 1, 2, 3], [9, 5, 1, 2]])
    True
    >>> isToeplitzMatrix([[1, 2], [2, 2]])
    False
    >>> isToeplitzMatrix([[11,74,0,93],[40,11,74,7]])
    False
    """
    return all(r == 0 or c == 0 or matrix[r - 1][c - 1] == val for r, row in enumerate(matrix) for c, val in enumerate(row))


# %% 899. https://leetcode.com/problems/orderly-queue/
# A total troll problem that was spoiled a bit by Github Copilot. I thought it hinted an obviously wrong solution, but on further thought
# it made me realize that the solution is dead simple.
def orderlyQueue(s: str, k: int) -> str:
    """
    Examples:
    >>> orderlyQueue("cba", 1)
    'acb'
    >>> orderlyQueue("baaca", 3)
    'aaabc'
    >>> orderlyQueue("baaca", 1)
    'aacab'
    >>> orderlyQueue("baaca", 2)
    'aaabc'
    >>> orderlyQueue("baaca", 4)
    'aaabc'
    >>> orderlyQueue("badaca", 2)
    'aaabcd'
    >>> orderlyQueue("badacadeff", 3)
    'aaabcddeff'
    """
    if k == 1:
        return min(s[i:] + s[:i] for i in range(len(s)))

    return "".join(sorted(s))


# %% 901. https://leetcode.com/problems/online-stock-span/
# Lessons learned:
# - this uses a monotonically decreasing stack (in the first coordinate); the complexity of an MDS is linear
# - we don't need the full stack, so we compress into (value, number of elements before a bigger value); not sure how to analyze the complexity
class StockSpanner:
    """
    Examples:
    >>> obj = StockSpanner()
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


# %% 947. https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/
def removeStones(stones: list[list[int]]) -> int:
    """
    This is O(n**2) in len(stones). Unnecesary, but C- connected components doesn't make up for it.

    Examples:
    >>> removeStones([[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]])
    5
    >>> removeStones([[0,0],[0,2],[1,1],[2,0],[2,2]])
    3
    >>> removeStones([[0,0]])
    0
    >>> removeStones([[0,0],[0,1],[1,1]])
    2
    >>> removeStones([[0,1],[1,0]])
    0
    """

    mat = csr_matrix((len(stones), len(stones)), dtype=bool)
    for i, j in permutations(range(len(stones)), 2):
        if stones[i][0] == stones[j][0] or stones[i][1] == stones[j][1]:
            mat[i, j] = True
            mat[j, i] = True

    n_components = connected_components(mat, directed=False, return_labels=False)
    return len(stones) - n_components


def removeStones2(stones: list[list[int]]) -> int:
    """
    Examples:
    >>> removeStones2([[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]])
    5
    >>> removeStones2([[0,0],[0,2],[1,1],[2,0],[2,2]])
    3
    >>> removeStones2([[0,0]])
    0
    >>> removeStones2([[0,0],[0,1],[1,1]])
    2
    >>> removeStones2([[0,1],[1,0]])
    0
    """
    rows = defaultdict(list)
    cols = defaultdict(list)

    for i, (r, c) in enumerate(stones):
        rows[r].append(i)
        cols[c].append(i)

    seen = set()

    def dfs(i: int) -> None:
        seen.add(i)
        for j in rows[stones[i][0]] + cols[stones[i][1]]:
            if j not in seen:
                dfs(j)

    n_components = 0
    for i in range(len(stones)):
        if i not in seen:
            dfs(i)
            n_components += 1

    return len(stones) - n_components


# %% 990. https://leetcode.com/problems/satisfiability-of-equality-equations/
def equationsPossible(equations: list[str]) -> bool:
    """
    Examples:
    >>> assert equationsPossible(["a==b", "b!=a"]) is False
    >>> assert equationsPossible(["x==y", "z==w", "y==z", "a==b", "d==e", "f==g", "e==f", "w==x", "c==d", "b==d", "g!=x"]) is True
    >>> assert equationsPossible(["x==y", "z==w", "y==z", "a==b", "d==e", "f==g", "e==f", "w==x", "c==d", "b==d", "g!=x", "a==z"]) is False
    >>> assert equationsPossible(["x==a", "w==b", "z==c", "a==b", "b==c", "c!=x"]) is False
    >>> assert equationsPossible(["a==b", "c==e", "b==c", "a!=e"]) is False
    >>> assert equationsPossible(["a==b", "e==c", "c==b", "a!=e"]) is False
    >>> assert equationsPossible(["a==b", "e==c", "c==b", "a!=e"]) is False
    >>> assert equationsPossible(["a==b", "e==c", "b==c", "a!=e"]) is False
    """
    equivalence_classes = []
    for x, eq, _, y in equations:
        if eq == "=":
            if x == y:
                continue
            for equivalence_class in equivalence_classes:
                if x in equivalence_class or y in equivalence_class:
                    equivalence_class |= {x, y}
                    break
            else:
                equivalence_classes.append({x, y})

    while True:
        merged = False
        for i in range(len(equivalence_classes)):
            for j in range(i + 1, len(equivalence_classes)):
                if len(equivalence_classes[i] & equivalence_classes[j]) > 0:
                    equivalence_classes[i] |= equivalence_classes[j]
                    equivalence_classes.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break

    for x, eq, _, y in equations:
        if eq == "!":
            if x == y:
                return False
            for equivalence_class in equivalence_classes:
                if x in equivalence_class and y in equivalence_class:
                    return False
    return True


# You can use Disjoint Set Union (aka Union Find): https://cp-algorithms.com/data_structures/disjoint_set_union.html
# I prefer the iterative version, but recursive could work for small inputs.
def equationsPossible2(equations: list[str]) -> bool:
    """
    Examples:
    >>> assert equationsPossible2(["a==b", "b!=a"]) is False
    >>> assert equationsPossible2(["x==y", "z==w", "y==z", "a==b", "d==e", "f==g", "e==f", "w==x", "c==d", "b==d", "g!=x"]) is True
    >>> assert equationsPossible2(["x==y", "z==w", "y==z", "a==b", "d==e", "f==g", "e==f", "w==x", "c==d", "b==d", "g!=x", "a==z"]) is False
    >>> assert equationsPossible2(["x==a", "w==b", "z==c", "a==b", "b==c", "c!=x"]) is False
    >>> assert equationsPossible2(["a==b", "c==e", "b==c", "a!=e"]) is False
    >>> assert equationsPossible2(["a==b", "e==c", "c==b", "a!=e"]) is False
    >>> assert equationsPossible2(["a==b", "e==c", "c==b", "a!=e"]) is False
    >>> assert equationsPossible2(["a==b", "e==c", "b==c", "a!=e"]) is False
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        y = x
        while True:
            if y != parent[y]:
                y = parent[y]
                continue
            break
        parent[x] = y
        return parent[x]

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


# %% 1047. https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/
def remove_duplicates(s: str) -> str:
    """
    Examples:
    >>> remove_duplicates("abbaca")
    'ca'
    >>> remove_duplicates("aaaaaaaa")
    ''
    """
    stack = []
    for c in s:
        if stack and c == stack[-1]:
            stack.pop()
        else:
            stack.append(c)
    return "".join(stack)


# %% 1293. https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/
# Lessons learned:
# - You don't need a dictionary of best distances, just a set of visited nodes (since any first visit to a node is the best).
# - You don't need a priority queue, just a queue.


State = namedtuple("State", "steps k i j")


def shortestPath(grid: list[list[int]], k: int) -> int:
    """
    Examples:
    >>> shortestPath([[0,0,0],[1,1,0],[0,0,0],[0,1,1],[0,0,0]], 1)
    6
    >>> shortestPath([[0,1,1],[1,1,1],[1,0,0]], 1)
    -1
    >>> shortestPath([[0,0,0,0,0,0,0,0,0,0],[0,1,1,1,1,1,1,1,1,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,1,1,1,1,1,1,1],[0,1,0,0,0,0,0,0,0,0],[0,1,1,1,1,1,1,1,1,0],[0,1,0,0,0,0,0,0,0,0],[0,1,0,1,1,1,1,1,1,1],[0,1,0,1,1,1,1,0,0,0],[0,1,0,0,0,0,0,0,1,0],[0,1,1,1,1,1,1,0,1,0],[0,0,0,0,0,0,0,0,1,0]], 1)
    20
    """
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


# %% 1323. https://leetcode.com/problems/maximum-69-number/
# Lessons learned:
# - Converting to a string and using replace is surprisingly fast.
# - Just need to accept that Python string built-ins are in C-land.
def maximum69Number(num: int) -> int:
    """
    Examples:
    >>> maximum69Number(9669)
    9969
    >>> maximum69Number(9996)
    9999
    >>> maximum69Number(9999)
    9999
    """
    for i in range(math.floor(math.log10(num)) + 1, -1, -1):
        if num // 10**i % 10 == 6:
            return num + 3 * 10**i
    return num


def maximum69Number2(num: int) -> int:
    return int(str(num).replace("6", "9", 1))


# %% 1544. https://leetcode.com/problems/make-the-string-great/
def makeGood(s: str) -> str:
    """
    Examples:
    >>> makeGood("leEeetcode")
    'leetcode'
    >>> makeGood("abBAcC")
    ''
    >>> makeGood("s")
    's'
    """
    stack = []
    for c in s:
        if stack and stack[-1].lower() == c.lower() and stack[-1] != c:
            stack.pop()
        else:
            stack.append(c)
    return "".join(stack)


# %% 1680. https://leetcode.com/problems/concatenation-of-consecutive-binary-numbers/
def concatenatedBinary(n: int) -> int:
    """
    Examples:
    >>> concatenatedBinary(1)
    1
    >>> concatenatedBinary(3)
    27
    >>> concatenatedBinary(12)
    505379714
    """
    M = 10**9 + 7
    total = 1
    for i in range(2, n + 1):
        total = ((total << math.floor(math.log2(i)) + 1) + i) % M

    return total


# %% 1706. https://leetcode.com/problems/where-will-the-ball-fall/
def findBall(grid: list[list[int]]) -> list[int]:
    """
    Examples:
    >>> findBall([[-1]])
    [-1]
    >>> findBall([[1,1,1,-1,-1],[1,1,1,-1,-1],[-1,-1,-1,1,1],[1,1,1,1,-1],[-1,-1,-1,-1,-1]])
    [1, -1, -1, -1, -1]
    >>> findBall([[1,1,1,1,1,1]])
    [1, 2, 3, 4, 5, -1]
    >>> findBall([[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1]])
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


# %% 2131. https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/
def longestPalindrome(words: list[str]) -> int:
    """
    Examples:
    >>> longestPalindrome(["ab","ba","aa","bb","cc"])
    6
    >>> longestPalindrome(["ab","ba","cc","ab","ba","cc"])
    12
    >>> longestPalindrome(["aa","ba"])
    2
    >>> longestPalindrome(["ba", "ce"])
    0
    >>> longestPalindrome(["lc","cl","gg"])
    6
    >>> longestPalindrome(["ab","ty","yt","lc","cl","ab"])
    8
    >>> longestPalindrome(["cc","ll","xx"])
    2
    """
    d = Counter()

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


# %% 2269. https://leetcode.com/problems/find-the-k-beauty-of-a-number/
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


# %%
