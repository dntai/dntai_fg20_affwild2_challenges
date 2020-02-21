#!/bin/sh
conda env create -n cur37 -f cur37_pip.yml
source activate cur37
python -m ipykernel install --user --name cur37 --display-name "cur37"
conda install -y -c defaults protobuf libprotobuf
conda install -y glog leveldb
source deactivate
