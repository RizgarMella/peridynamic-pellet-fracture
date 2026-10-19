#!/bin/bash
# Build pellet_pd with the build id (sha256 of the source) embedded in every output.
set -euo pipefail
cd "$(dirname "$0")"
BUILD=$(sha256sum pellet_pd.cpp | cut -c1-16)
CXX=${CXX:-g++}
$CXX -O3 -march=${MARCH:-native} -fopenmp -std=c++17 -DPD_BUILD="\"$BUILD\"" pellet_pd.cpp -o pellet_pd
echo "built pellet_pd (build $BUILD)"
