#!/bin/bash
#
# Pulish all notebooks to mxnet.
#
# Assume your mxnet repo is available at ~/mxnet
#
# Then the following command will convert all notebooks into markdown files and
# put them at ~/mxnet/docs/tutorials/gulon.
#
# ./publish.sh ~/mxnet
#
# Later you can create a PR to mxnet to merge these markdown files. Once down,
# they will show up at http://mxnet.io/tutorials/

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

if [ "$#" -ne 1 ]; then
    echo "Pulish all notebooks to mxnet."
    echo "Usage:"
    echo ""
    echo "  $0 path_to_your_mxnet"
    echo ""
    echo "e.g. $0 ~/mxnet"
    exit 1
fi

MXNET_HOME=$1


echo "Pulish all notebooks into ${MXNET_HOME}/docs/tutorials/gluon"

for f in ${NOTEBOOK_DIR}/*.ipynb; do
    python ${MXNET_HOME}/tools/ipynb2md.py $f
done

for f in ${NOTEBOOK_DIR}/*.md; do
    if [[ $f == *"README.md"* ]]; then
        continue
    fi
    mv $f ${MXNET_HOME}/docs/tutorials/gluon/
done

echo "DONE!"
echo "Now either run make docs in ${MXNET_HOME} or create a PR to merge the changes"
