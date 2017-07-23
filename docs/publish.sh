#!/bin/bash
#
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

cd ${NOTEBOOK_DIR}

make html

# make html latex && make -C _build/latex

# cp _build/latex/*pdf _build/html/

cd _build && zip -r html.zip html

scp html.zip ubuntu@mxnet.io:/gluon-docs/

ssh ubuntu@mxnet.io "cd /gluon-docs && unzip html.zip && rm -rf latest && mv html latest"
