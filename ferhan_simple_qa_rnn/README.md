## Directions - Relation Prediction Model

- Download SimpleQuestions data from this [link](https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz) and put it in directory "data/SimpleQuestions_v2/"
- Run the training script with the following commands. Please check out args.py file to see the different commands available:
```
cd relation_prediction
python train.py
python train.py --cuda
python train.py --cuda --rnn_type gru
```
