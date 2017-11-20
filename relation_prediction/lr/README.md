## Logistic Regression for Relation Prediction

### For tf-idf features
```
python lr_tfidf.py --help
```
- If you have a trained model, then use:
```
python lr_tfidf.py --trained_model path/to/model/pickle
```

### For embeddings + top 300 relation features
- We average the word embeddings (GloVe) of the words in the sentence and use the top 300 relation words.
```
python lr_embeddings_rel.py --help
```
- If you have a trained model, then use:
```
python lr_embeddings_rel.py --trained_model path/to/model/pickle
```

### Results
- embeddings + 300 top rel words:
```
Testing on valid data...
Accuracy on valid dataset: 70.63162747810051
Hits: 1, Retrieved: 7660, Total: 10845, RetrievalRate: 70.63162747810051
Hits: 3, Retrieved: 9295, Total: 10845, RetrievalRate: 85.70769940064545
Hits: 5, Retrieved: 9666, Total: 10845, RetrievalRate: 89.12863070539419

Testing on test data...
Accuracy on test dataset: 70.8442845944575
Hits: 1, Retrieved: 15364, Total: 21687, RetrievalRate: 70.8442845944575
Hits: 3, Retrieved: 18559, Total: 21687, RetrievalRate: 85.57661271729607
Hits: 5, Retrieved: 19352, Total: 21687, RetrievalRate: 89.23318116844193
```

- tfidf on 1-gram and 2-gram:
```
Testing on valid data...
Accuracy on valid set: 72.43643088303283
Hits: 1, Retrieved: 7834, Total: 10815, RetrievalRate: 72.43643088303283
Hits: 3, Retrieved: 9163, Total: 10815, RetrievalRate: 84.72491909385113
Hits: 5, Retrieved: 9477, Total: 10815, RetrievalRate: 87.62829403606102

Testing on test data...
Accuracy on test set: 71.99648684879583
Hits: 1, Retrieved: 15575, Total: 21633, RetrievalRate: 71.99648684879583
Hits: 3, Retrieved: 18347, Total: 21633, RetrievalRate: 84.8102436093006
Hits: 5, Retrieved: 18975, Total: 21633, RetrievalRate: 87.71321592012204
```
