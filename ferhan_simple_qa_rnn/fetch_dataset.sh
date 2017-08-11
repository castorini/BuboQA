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

echo -e "\n\nCreate modified dataset..."
python scripts/modify_dataset.py -d data/SimpleQuestions_v2 -o data/SimpleQuestions_v2_modified

echo -e "\n\nDONE!"