#!/bin/bash
#
# Build and publish all docs into Pulish all notebooks to mxnet.
set -x
set -e

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
cd ${NOTEBOOK_DIR}

# install a gpu version
# sed -i.bak s/mxnet/mxnet-cu90/g environment.yml

# prepare the env
conda env update -f environment.yml
source activate gluon_docs

make html

rm -rf ~/www/latest
mv _build/html ~/www/latest
