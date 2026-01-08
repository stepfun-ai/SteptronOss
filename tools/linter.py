#!/usr/bin/env python3
import pathlib
from glob import glob
from os.path import exists, join
from sys import argv


def recursively_lint_files(files):
    """Recursively lint all python files in chosen subdirectories of megatron-lm"""

    import black
    import isort

    # get all python file paths from top level directory
    tools_dir = pathlib.Path(__file__).parent.absolute()
    repo_dir = tools_dir.parent
    if not files:
        all_py_paths = glob(str(repo_dir) + "/**/*.py")
    else:
        all_py_paths = filter(
            lambda x: x.endswith(".py"),
            filter(exists, [join(repo_dir, x) for x in files]),
        )

    print("Linting the following: ")
    all_good = True
    for py_path in all_py_paths:
        orig_file = open(py_path, "r").read()
        isort.file(py_path)
        black.format_file_in_place(
            pathlib.Path(py_path),
            fast=False,
            mode=black.Mode(),
            write_back=black.WriteBack.YES,
        )
        modified = open(py_path, "r").read() != orig_file
        if modified:
            all_good = False
    return all_good


if __name__ == "__main__":
    recursively_lint_files(argv[1:])
