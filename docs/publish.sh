#!/bin/bash
#
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd ${NOTEBOOK_DIR}

# need to run `conda env create -f environment.yml' first

source activate gluon_docs

pip install --upgrade --pre mxnet
pip show mxnet

make html

rm -rf /gluon-docs/latest
mv _build/html /gluon-docs/latest
