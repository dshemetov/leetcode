"""
Convert between a monofile and a multifile for LeetCode problems.

Automating indecision by learning about the ast module

    https://docs.python.org/3/library/ast.html#ast-helpers

The comments are currently not preserved. We also don't track dependencies, so
just dump all the dependencies in every file for now. This is a good start, but
we can probably do better. Also for some reason function docstrings inherit an
extra newline in every line.

TODO:
- Unclear how to preserve comments. How does Black do it?
https://github.com/psf/black/blob/6af7d1109693c4ad3af08ecbc34649c232b47a6d/src/black/comments.py#L34
- Track dependencies and only include the necessary imports in each file.
  - Dumb workaround: run Ruff on the file after writing it.
- Fix the docstring issue.
"""

import ast
import re
from pathlib import Path

import typer

app = typer.Typer(name="LeetCode Solution Converter", chain=True)


@app.command("to-monofile")
def convert_to_monofile() -> None:
    """Convert a multifile to a monofile.

    Currently just stitches together all the files in the problems directory.
    """
    monofile_docstring = """
\"\"\"
Python solutions to LeetCode problems.

Keeping it as a monofile because there's something funny about it.
\"\"\"
    """
    with open("python/problems/monofile.py", "w") as f:
        f.write(monofile_docstring)
        for file in sorted(Path("python/problems/").glob("*.py")):
            f.write("\n\n")
            f.write(file.read_text())


@app.command("to-multifile")
def convert_to_multifile() -> None:
    """Convert a monofile to a multifile.

    Assumes that all the code use by a problem precedes the problem function
    definition.
    """
    file_text = "\n".join(open("python/problems.py").readlines())
    tree = ast.parse(file_text)
    import_statements = []
    problem_chunks = []
    stack = []
    for x in tree.body[1:]:
        if isinstance(x, ast.Import):
            import_statements.append(x)
        elif isinstance(x, ast.ImportFrom):
            import_statements.append(x)
        elif isinstance(x, ast.FunctionDef) and re.match(r"p(\d+)", x.name):
            stack.append(x)
            problem_chunks.append(stack)
            stack = []
        else:
            stack.append(x)

    Path("python/problems/").mkdir(exist_ok=True)
    for problem_chunk in problem_chunks:
        import_text = "\n".join(ast.unparse(chunk) for chunk in import_statements)
        problem_text = "\n\n".join(ast.unparse(chunk) for chunk in problem_chunk)
        problem_number = re.search(r"p(\d+)", problem_chunk[-1].name).group(1)

        with open(f"python/problems/p{int(problem_number):04d}.py", "w") as f:
            f.write(import_text)
            f.write("\n\n")
            f.write(problem_text)


if __name__ == "__main__":
    app()
