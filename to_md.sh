#!/bin/bash

for file in P*.ipynb; do
    notedown "$file" --to markdown > "`basename "$file" .ipynb`.md"
done
