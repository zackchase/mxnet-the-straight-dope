#!/bin/bash

for file in P*.md; do
    notedown "$file" > "`basename "$file" .md`.ipynb"
done
