#!/bin/bash
#
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x
set -e

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd ${NOTEBOOK_DIR}

# prepare the env
conda env update -f environment.yml 
source activate gluon_docs

# install a gpu version
pip uninstall mxnet
pip install --pre mxnet-cu80
pip show mxnet

make html

rm -rf ~/www/latest
mv _build/html ~/www/latest
