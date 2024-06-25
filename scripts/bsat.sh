#!/bin/bash

# Script to start birch with the correct environment variables
# TODO: Probably should just remove any dependence on environment variables

export BIRCH_PLUGINS=plugins
LD_LIBRARY_PATH=plugins

pushd bin
./bsat $@
popd
