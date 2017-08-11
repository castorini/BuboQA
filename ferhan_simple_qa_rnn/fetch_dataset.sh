#!/bin/bash

# download dataset and put it in data directory
mkdir data
pushd data

echo -e "\n\nDownloading SimpleQuestions dataset..."
wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz

echo -e "\n\nUnzipping SimpleQuestions dataset..."
tar -xvzf SimpleQuestions_v2.tgz

echo -e "\n\nDownloading the names file..."
wget https://www.dropbox.com/s/yqbesl07hsw297w/FB5M.name.txt

popd

echo -e "\n\nTrimming the names file for subset 2M..."
python scripts/trim_names.py -s data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt -n data/FB5M.name.txt -o data/names.trimmed.2M.txt

echo -e "\n\nTrimming the names file for subset 5M..."
python scripts/trim_names.py -s data/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt -n data/FB5M.name.txt -o data/names.trimmed.5M.txt

echo -e "\n\nCreate modified dataset..."
python scripts/modify_dataset.py -d data/SimpleQuestions_v2 -o data/SimpleQuestions_v2_modified

echo -e "\n\nDONE!"