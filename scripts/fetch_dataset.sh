#!/bin/bash

# download dataset and put it in data directory
mkdir data

echo "Downloading SimpleQuestions dataset...\n"
wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz

echo "\n\nUnzipping SimpleQuestions dataset...\n"
tar -xvzf SimpleQuestions_v2.tgz

mv SimpleQuestions_v2 data/
rm SimpleQuestions_v2.tgz

echo "Downloading Embeddings...\n"
wget http://ocp59jkku.bkt.clouddn.com/sq_glove300d.pt
mv sq_glove300d.pt data/

echo "Downloading Features...\n"
wget http://ocp59jkku.bkt.clouddn.com/feature4lr.zip
unzip feature4lr.zip -d lr_glove_rel_features/
mv lr_glove_rel_features/ data/
rm feature4lr.zip

echo "Downloading Mapping to wiki...\n"
wget http://ocp59jkku.bkt.clouddn.com/fb2w.nt
mv fb2w.nt data/

#echo "\n\nDownloading the augmented FB2M graph and names file...\n"
#wget https://www.dropbox.com/s/yqbesl07hsw297w/FB5M.name.txt
#wget https://www.dropbox.com/s/8tcagdi2iq8q0w5/fb-2M-augmented.txt

mkdir data/freebase_names
echo "\n\nDownloading the names file...\n"
#wget https://www.dropbox.com/s/yqbesl07hsw297w/FB5M.name.txt
wget http://ocp59jkku.bkt.clouddn.com/FB5M.name.txt

mv FB5M.name.txt data/freebase_names/

echo "\n\nTrimming the names file for subset 2M...\n"
python scripts/trim_names.py -s data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt -n data/freebase_names/FB5M.name.txt -o data/freebase_names/names.trimmed.2M.txt

#echo "\n\nTrimming the names file for subset 5M...\n"
#python scripts/trim_names.py -s data/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt -n data/freebase_names/FB5M.name.txt -o data/freebase_names/names.trimmed.5M.txt

# creating the name index
mkdir indexes

echo "Create the names map index for 2M-freebase-subset...\n"
python scripts/create_index_names.py -n data/freebase_names/names.trimmed.2M.txt -p indexes/names_2M.pkl

# echo "\n\nCreate the names map index for 5M-freebase-subset...\n"
# python scripts/create_index_names.py -n data/freebase_names/names.trimmed.5M.txt -p indexes/names_5M.pkl

echo "\n\nCreate processed, augmented dataset...\n"
python scripts/augment_process_dataset.py -d data/SimpleQuestions_v2 -i indexes/names_2M.pkl -o data/processed_simplequestions_dataset

# get the lineids
awk '{ print $1 }' data/processed_simplequestions_dataset/all.txt > data/processed_simplequestions_dataset/lineids_all.txt
awk '{ print $1 }' data/processed_simplequestions_dataset/train.txt > data/processed_simplequestions_dataset/lineids_train.txt
awk '{ print $1 }' data/processed_simplequestions_dataset/valid.txt > data/processed_simplequestions_dataset/lineids_valid.txt
awk '{ print $1 }' data/processed_simplequestions_dataset/test.txt > data/processed_simplequestions_dataset/lineids_test.txt

echo "\n\nDONE!"
