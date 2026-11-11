#!/bin/bash
# One-shot setup of the Python environment for the LMPC book examples on a Mac.
# Same setup Ugo verified (macOS arm64, Python 3.9.6, numpy 2.0.2, CasADi 3.7.2 from pip).
#
# Usage, from Terminal (works on any Mac, run it again any time - it is safe):
#     cd ~/Learning_Robust_MPC
#     bash setup_mac.sh
#
# What it does:
#   1. creates the environment ~/venv_lmpc (if it does not exist yet)
#   2. updates pip and installs the packages listed in requirements-lmpc.txt
#   3. registers the environment as a Jupyter kernel called "LMPC book (Python 3.9)",
#      so VS Code and JupyterLab show it in their kernel list
#   4. runs a 20-second speed test (expected: about 0.1-0.2 s per LMPC step)
set -e
cd "$(dirname "$0")"
VENV="$HOME/venv_lmpc"

# Python 3.9.6 is the one Apple installs with the Xcode command line tools
PY=/usr/bin/python3
if ! "$PY" -c 'import sys' 2>/dev/null; then
    echo "Apple's Python is missing. Run:  xcode-select --install   then run this script again."
    exit 1
fi

# Don't let a conda environment interfere
unset PYTHONPATH CONDA_PREFIX

if [ ! -x "$VENV/bin/python" ]; then
    echo "== Creating $VENV"
    "$PY" -m venv "$VENV"
fi

echo "== Updating pip"
"$VENV/bin/python" -m pip install -q --upgrade pip

echo "== Installing packages"
"$VENV/bin/python" -m pip install -q -r requirements-lmpc.txt

echo "== Registering the Jupyter kernel"
"$VENV/bin/python" -m ipykernel install --user --name lmpc-book --display-name "LMPC book (Python 3.9)"

echo "== Speed test"
"$VENV/bin/python" Nominal_LMPC_Chapter/Ex2_LMPC_for_minimum_time_unicycle/timing_test.py

echo
echo "Done. In VS Code: open a notebook, click the kernel name at the top right,"
echo "choose 'Select Another Kernel' -> 'Jupyter Kernel' -> 'LMPC book (Python 3.9)'."
