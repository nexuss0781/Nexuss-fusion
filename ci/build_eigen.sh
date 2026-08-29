#!/usr/bin/env bash
set -euo pipefail

# Build the optional C++/Eigen fallback kernel as nexuss_fusion/backend/_eigen_native*.so
# Usage: ci/build_eigen.sh [eigen-include-dir]

PYTHON="${PYTHON:-python3}"
EIGEN_INC="${EIGEN_INC:-${1:-}}"

if [[ -z "$EIGEN_INC" ]]; then
  for cand in /usr/include/eigen3 /usr/local/include/eigen3; do
    if [[ -f "$cand/Eigen/Dense" ]]; then
      EIGEN_INC="$cand"
      break
    fi
  done
fi
if [[ -z "$EIGEN_INC" || ! -f "$EIGEN_INC/Eigen/Dense" ]]; then
  echo "Eigen headers not found; install libeigen3-dev or pass the include dir." >&2
  exit 1
fi

PYBIND_INC="$("$PYTHON" -c 'import pybind11; print(pybind11.get_include())')"
EXT_SUFFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="nexuss_fusion/backend/_eigen_native${EXT_SUFFIX}"

g++ -O3 -march=native -shared -fPIC \
  -I"$PYBIND_INC" -I"$EIGEN_INC" \
  cpp/eigen_kernels.cpp \
  -o "$OUT"

"$PYTHON" - <<PY
import sys
sys.path.insert(0, ".")
from nexuss_fusion.backend import get_backend
assert get_backend("eigen").name == "eigen", "eigen backend not selected after build"
b = get_backend("eigen")
print("eigen backend OK:", b.name)
PY