"""
Convert between a monofile and a multifile for LeetCode problems.

Automating indecision.

TODO:
- https://docs.python.org/3/library/ast.html#ast-helpers
"""

import ast


def convert_to_multifile(monofile: str, multifile: str) -> None:
    """
    Convert a monofile to a multifile.
    """
    ...


file_text = "\n".join(open("python/problems.py").readlines())
tree = ast.parse(file_text)
ast.dump(tree)

tree = ast.parse("print('Hello, world!')")
ast.dump(tree)
