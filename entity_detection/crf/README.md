## Entity Detection with Stanford NERTagger

### Quick Start

```
bash auto_run.sh
```
This script will automatically download the Stanford NER jar and run it on the dataset and then output2query.py takes that as input and converts to query text. You will get the query text on the *query_text* directory.


### Results

| Dataset | Precision | Recall | F1 |
|:-------:|:---------:|:------:|:--:|
| Valid   | 90.88 | 89.98 | 90.39 |
| Test    | 90.75 | 89.80 | 90.27 | 