## Relation Prediction Model

- Download and extract SimpleQuestions dataset by running the script:
```
bash fetch_dataset.sh 
```

- Run the training script with the following commands. Please check out args.py file to see the different commands available:
```
cd relation_prediction
python train.py
python train.py --cuda
python train.py --cuda --rnn_type gru
```
