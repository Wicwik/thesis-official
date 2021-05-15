#!/bin/bash

git submodule update --init --recursive
conda env create

export PYTHONPATH="$PWD/01_generate_samples/stylegan2:$PWD/01_generate_samples/src:$PWD/03_regress_feature_axis/src"
