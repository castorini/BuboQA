## Simple RNN model

- Download and extract SimpleQuestions dataset by running the script:
```
bash fetch_dataset.sh 
```

- Create the indexes for the 2M Freebase subset with this script:
```
bash create_indexes.sh
```

- Then go to the 'entity_linking' directory and start the Jupyter notebook to play with the linking phase:
```
cd entity_linking
jupyter notebook
```

- You can also directly run the linking script.
```
python entity_linking.py -t ../data/SimpleQuestions_v2_modified/test.txt --index_ent ../indexes/entity_2M.pkl --index_reach ../indexes/reachability_2M.pkl \
    --index_names ../indexes/names_2M.pkl --ent_result ../entity_detection/query-text/test.txt \
    --rel_result ../relation_prediction/results/main-test-results.txt --output ./results
```

- To evaluate results:
```
cd entity_linking
python evaluate.py -g ../data/SimpleQuestions_v2_modified/test.txt  -p results/linking-results.txt
```