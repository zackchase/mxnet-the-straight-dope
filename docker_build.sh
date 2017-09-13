#!/bin/bash
set -e
set -x
docker build --rm -f Dockerfile.build.ubuntu-17.04 -t mxnet-the-straight-dope .
docker run --rm mxnet-the-straight-dope make $@
