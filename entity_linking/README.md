## Enity Linking

### Quick Start

- You can run linking like this:
```
python entity_linking --help
python entity_linking --model_type crf
```
- You must provide the model type. Change the query directory according to the model type.

### Results

#### Entity Linking with BiLSTM

| Dataset | 1@R | 3@R | 5@R | 10@R | 20@R | 50@R | 100@R |
|:-------:|:---:|:---:|:---:|:----:|:----:|:----:|:-----:|
|  Test   |65.002542|76.429529|79.790135|83.127629|85.753247|88.383488|90.315721|


#### Entity Linking with CRF

| Dataset | 1@R | 3@R | 5@R | 10@R | 20@R | 50@R | 100@R |
|:-------:|:---:|:---:|:---:|:----:|:----:|:----:|:-----:|
|  Test   |63.731336|74.728424|78.310914|81.532843|83.996672|86.742477|88.781029|

