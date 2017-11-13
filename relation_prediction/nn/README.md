## Neural Network for Relation Prediction

### Quick Start with CNN

```
python train.py --fix_embed --relation_prediction_mode CNN
or 
python train.py --fix_embed --relation_prediction_mode CNN --no_cuda
```

### Quick Start with GRU

```
python train.py --fix_embed --relation_prediction_mode GRU
or 
python train.py --fix_embed --relation_prediction_mode GRU --no_cude
```

The model file will be saved in *saved_checkpoints*.
You can use
```
python top_retrieval.py --relation_prediction_mode CNN --trained_model path_to_file --hits 5
or 
python top_retrieval.py --relation_prediction_mode GRU --trained_model path_to_file --hits 5
```

