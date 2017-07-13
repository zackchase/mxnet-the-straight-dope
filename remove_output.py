#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Remove output from existing Jupyter Notebooks.
Modified from remove_output by Minrk, damianavila, gabraganca.
References:
    [0]: https://github.com/jupyter/nbformat
    [1]: http://nbformat.readthedocs.org/en/latest/index.html
    [2]: http://blog.jupyter.org/2015/04/15/the-big-split/
"""

import sys
import io
import os
import argparse
import nbformat


def remove_outputs(nb):
    """Removes the outputs cells for a jupyter notebook."""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []


def clear_notebook(old_ipynb, new_ipynb):
    with io.open(old_ipynb, 'r') as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    remove_outputs(nb)

    with io.open(new_ipynb, 'w', encoding='utf8') as f:
        nbformat.write(nb, f, nbformat.NO_CONVERT)


def main():
    parser = argparse.ArgumentParser(
        description="Remove output cells of Jupyter Notebooks."
    )

    parser.add_argument(
        "notebook", nargs="+", help="The notebook to be cleared."
    )
    parser.add_argument(
        "-o", "--output", dest="output_filename",
        help="Writes to FILE. If not set, it will rewrite.", metavar="FILE"
    )

    args = parser.parse_args()
    nbs = len(args.notebook)

    if nbs > 1:
       for old_ipynb in args.notebook:
            clear_notebook(old_ipynb, old_ipynb)
    else:
        old_ipynb = args.notebook[0]
        # base, ext = os.path.splitext(fname)
        # new_ipynb = "%s_removed%s" % (base, ext)

        try:
            new_ipynb = os.path.splitext(args.output_filename)[0] + '.ipynb'
        except AttributeError:
            new_ipynb = old_ipynb

        clear_notebook(old_ipynb, new_ipynb)


if __name__ == '__main__':
    main()
