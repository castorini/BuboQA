## Relation Prediction

The pipeline should be like this:
- go into any of the folders corresponding to a relation prediction model
- check the parameters in args.py
- train the model
```
cd nn
python train.py --fix_embed --relation_prediction_mode CNN
or 
python train.py --fix_embed --relation_prediction_mode CNN --no_cuda
```
- dump the best model, gets saved to 'saved_checkpoints' by default

Then:
- use the best model to predict on the validation/test set and get the top 5 results for relation label
```
python top_retrieval.py --relation_prediction_mode CNN --trained_model path_to_file --hits 5
```
- the output of that will in the 'results' directory






## Results

### GRU

| Dataset | 1@R | 3@R | 5@R |
|:-------:|:---:|:---:|:---:|
| Valid | 82.27 | 93.86 | 95.91 |
| Test  | 81.77 | 93.79 | 95.74 |

### CNN

| Dataset | 1@R | 3@R | 5@R |
|:-------:|:---:|:---:|:---:|
|  Valid  |82.82|93.56|95.89|
|  Test   |82.12|93.65|95.66|

### LR(tf-idf)

| Dataset | 1@R | 3@R | 5@R |
|:-------:|:---:|:---:|:---:|
|  Valid  |72.44|84.72|87.63|
|   Test  |72.00|84.81|87.71|

### LR(glove+rel)

| Dataset | 1@R | 3@R | 5@R |
|:-------:|:---:|:---:|:---:|
|  Valid  |74.71|88.93|92.16|
|  Test   |74.50|88.64|92.02|
