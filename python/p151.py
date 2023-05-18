# %% 151. Reverse Words In A String https://leetcode.com/problems/reverse-words-in-a-string/
from array import array


# Lesson learned:
# - Python string built-ins are fast.
# - The follow-up asks: can you use only O(1) extra space? Here is the trick:
#   first reverse the whole string and then reverse each word. Reversing each
#   requires keeping track of two pointers: the start of a word and the end of a
#   word (terminated by a space).
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
    return " ".join(s.split()[::-1])


def reverseWords2(s: str) -> str:
    """
    Examples:
    >>> reverseWords2("the sky is blue")
    'blue is sky the'
    >>> reverseWords2("  hello world!  ")
    'world! hello'
    >>> reverseWords2("a good   example")
    'example good a'
    >>> reverseWords2("  Bob    Loves  Alice   ")
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
