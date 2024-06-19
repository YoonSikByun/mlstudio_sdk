#!/bin/bash
# https://packaging.python.org/en/latest/tutorials/packaging-projects/
python3 -m build
pip3 install /opt/mlstudio/source/gitHub/mlstudio_sdk/dist/*.whl