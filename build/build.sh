#!/bin/bash
#
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x
set -e

# prepare the env
conda env update -f build/build.yml -n build_sd_tutorials
conda activate build_sd_tutorials

make html

# rm -rf build/data
# make pkg

# make pdf
# cp build/_build/latex/gluon_tutorials.pdf build/_build/html/

aws s3 sync --delete build/_build/html/ s3://gluon.mxnet.io/ --acl public-read
