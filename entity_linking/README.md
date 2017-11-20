## Cross Linking 

### Quick Start


### Results

#### Entity Linking with BiLSTM

| Dataset | 1@R | 3@R | 5@R | 10@R | 20@R | 50@R | 100@R |
|:-------:|:---:|:---:|:---:|:----:|:----:|:----:|:-----:|
|  Test   |65.002542|76.429529|79.790135|83.127629|85.753247|88.383488|90.315721|


#### Entity Linking with CRF

| Dataset | 1@R | 3@R | 5@R | 10@R | 20@R | 50@R | 100@R |
|:-------:|:---:|:---:|:---:|:----:|:----:|:----:|:-----:|
|  Test   |63.731336|74.728424|78.310914|81.532843|83.996672|86.742477|88.781029|


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


