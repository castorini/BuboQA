## Entity Detection

### Quick Start with RNN

- Go into *nn* directory

- Train the model 

```
python train.py --entity_detection_mode LSTM --fix_embed
or
python train.py --entity_detection_mode LSTM --fix_embed --no_cuda
```

to train the model.

The file should be saved in the *saved_checkpoints* directory.

- Test the model

```
python top_retrieval.py --trained_model [path/to/trained_model.pt] --entity_detection_mode LSTM
```

to test the model.
