## Retrieval-based model for simple question answering

- Please install the following Python 3 packages:
```
PyTorch (version 0.2.0)
torchtext (version 0.2.0)
NLTK 
NLTK data (tokenizers, stopwords list)
fuzzywuzzy
```

- Run the setup script. This takes a long time. It fetches dataset, other files, processes them and creates indexes:
```
sh setup.sh 
```


- There are four main components: entity detection, entity linking, relation prediction and evidence integration (read paper for more info)

- Each of these components have a directory and you can read the README file there on how to use them.

- entity_detection and relation_prediction can be run independently.
- entity_detection needs to be run before entity_linking.
- entity_linking and relation_prediction needs to be run before evidence_integration.
