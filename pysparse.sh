#!/bin/sh -e

. .venv/bin/activate
echo -e "\e[90m[PYINFO] Using python interpreter at $(which python)\e[0m"
exec python pysparse/py/bridge.py