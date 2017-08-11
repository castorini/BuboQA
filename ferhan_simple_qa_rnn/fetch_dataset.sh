#!/bin/bash

# download dataset and put it in data directory
mkdir data
pushd data
echo "Downloading SimpleQuestions dataset..."
wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz
echo "Unzipping SimpleQuestions dataset..."
tar -xvzf SimpleQuestions_v2.tgz
echo "Downloading the names file..."
wget https://www.dropbox.com/s/yqbesl07hsw297w/FB5M.name.txt
popd