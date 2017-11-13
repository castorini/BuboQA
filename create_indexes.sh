#!/bin/bash

# creating the indexes
mkdir indexes

echo "\n\nCreate the reachability index for 2M-freebase-subset...\n"
python scripts/create_index_reachability.py -s data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt -p indexes/reachability_2M.pkl

# echo "\n\nCreate the reachability index for 5M-freebase-subset...\n"
# python scripts/create_index_reachability.py -s data/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt -p indexes/reachability_5M.pkl

echo "\n\nCreate the inverted index for entity names for 2M-freebase-subset...\n"
python scripts/create_inverted_index_entity.py -n data/names.trimmed.2M.txt -p indexes/entity_2M.pkl

# echo "\n\nCreate the inverted index for entity names for 5M-freebase-subset...\n"
# python scripts/create_inverted_index_entity.py -n data/names.trimmed.5M.txt -p indexes/entity_5M.pkl

echo "\n\nDONE!"