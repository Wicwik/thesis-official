#!/bin/bash

git submodule update --init --recursive
conda env create

export PYTHONPATH="$PWD/01_generate_samples/stylegan2:$PWD/01_generate_samples/src:$PWD/03_regress_feature_axis/src"

sed -i 's/D_GLIBCXX_USE_CXX11_ABI=0/D_GLIBCXX_USE_CXX11_ABI=1/' 01_generate_samples/stylegan2/dnnlib/tflib/custom_ops.py
