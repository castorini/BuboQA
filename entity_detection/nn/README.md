## Entity Detection with RNNs

### Quick Start

```
python train.py --entity_prediction_mode GRU --fix_embed
or
python train.py --entity_prediction_mode LSTM --fix_embed --no_cuda
```

to train the model.

The file should be saved in the *saved_checkpoints* directory.

```
python main.py --trained_model [path/to/trained_model.pt]
```

to test the model.



### Results

#### LSTM

| Dataset | Precision | Recall | F1 | 
|:-------:|:---------:|:------:|:--:|
| Valid   | 92.92     | 93.23  | 93.08|
| Test    | 92.54     | 92.89  | 92.71|


#### GRU

| Dataset | Precision | Recall | F1 | 
|:-------:|:---------:|:------:|:--:|
| Valid   | 92.59     | 93.14  | 92.87|
| Test    | 92.15     | 92.80  | 92.47|


