for f in *.ipynb; do
    echo $f
    python remove_output.py $f
    jupyter nbconvert $f --to markdown
done
