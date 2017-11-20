## Evidence Integration

### Quick Start
```
python evidence_integration.py --help

python evidence_integration.py --ent_type crf --ent_path ../entity_linking/results/crf/test-h100.txt \
 --rel_type cnn --rel_path ../relation_prediction/nn/results/cnn/test.txt
```
- You must provide the arguments for the model type for entity linking (--ent_type) and the relation (--rel_type).
- Change the argument for the entity linking results directory (--ent_path) and the relation prediction result directory (--rel_path) to their model types.
- To run on the validation set:
```
python evidence_integration.py --data_path ../data/processed_simplequestions_dataset/valid.txt \
 --ent_type crf --ent_path ../entity_linking/results/crf/valid-h100.txt \
 --rel_type cnn --rel_path ../relation_prediction/nn/results/cnn/valid.txt
```


### Evidence Integration Results

#### Cross Linking with BiLSTMs + GRU

| Datast | 1@R | 2@R | 3@R |
|:------:|:---:|:---:|:---:|
|  Test  |73.147506|79.290898|81.149170|

#### Cross Linking with BiLSTMs + CNN

| Datast | 1@R | 2@R | 3@R |
|:------:|:---:|:---:|:---:|
|  Test  |73.189109|79.179956|81.042851|
|Test(-Heuristic)|54.768178| - | - |

#### Cross Linking with BiLSTMs + LR

| Dataset | 1@R | 2@R | 3@R |
|:-------:|:---:|:---:|:---:|
|   Test  |66.893172|73.041187|74.765405|


#### Cross Linking with CRF + GRU

| Dataset | 1@R | 2@R | 3@R |
|:-------:|:---:|:---:|:---:|
|  Test   |71.622059|77.751583|79.563630|


#### Cross Linking with CRF + CNN

| Dataset | 1@R | 2@R | 3@R |
|:-------:|:---:|:---:|:---:|
|  Test   |71.714510|77.659132|79.443443|


#### Cross Linking with CRF + LR

| Dataset | 1@R | 2@R | 3@R |
|:-------:|:---:|:---:|:---:|
|  Test   |65.543383|71.566588|73.244580|
|Test(-Heuristic)|47.543105|-|-|


