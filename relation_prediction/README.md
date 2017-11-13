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
| Valid | 82.293111 | 93.906611 | 95.968562 |
| Test  | 81.796330 | 93.750289 | 95.728748 |

### CNN

| Dataset | 1@R | 3@R | 5@R |
|:-------:|:---:|:---:|:---:|
|  Valid  |82.681461|93.666204|95.718909|
|  Test   |82.258586|93.625480|95.590071|

### LR

| Dataset | 1@R | 3@R | 5@R |
|:-------:|:---:|:---:|:---:|
|  Valid  |72.436431|84.724919|87.628294|
|   Test  |71.996487|84.810243|87.713216|

