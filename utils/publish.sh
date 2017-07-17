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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ane 1 ]
