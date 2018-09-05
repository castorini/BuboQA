# Simple Question Answering over Knowledge Graphs

This repo contains code for the following paper:

+ Salman Mohammed, Peng Shi, and Jimmy Lin. [Strong Baselines for Simple Question Answering over Knowledge Graphs with and without Neural Networks](http://aclweb.org/anthology/N18-2047). *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)*, pages 291-296, May 2018, New Orleans, Louisiana.

## Running the Code

Install the following Python 3 packages:

+ PyTorch (version 0.4.0)
+ torchtext (version 0.2.3)
+ NLTK 
+ NLTK data (tokenizers, stopwords list)
+ fuzzywuzzy

If you use PyTorch version 0.2.0, please checkout to 
```
commit 34a71f29192ed57f83d8576002f2b540de7d722f
```

Run the setup script. This takes a long time. It fetches dataset, other files, processes them and creates indexes:

```
sh setup.sh 
```

There are four main components to our formulation of the problem, as detailed in the paper: entity detection, entity linking, relation prediction and evidence integration. Each of these components is contained in a separate directory, with an associated README.

+ `entity_detection` and `relation_prediction` can be run independently.
+ `entity_detection` needs to be run before `entity_linking`.
+ `entity_linking` and `relation_prediction` needs to be run before `evidence_integration`.


## Running the Code with Docker (GPU, Ubuntu 16, Cuda 9.0 base)

- Make sure you have the Docker daemon running

- Build the image from Dockerfile
```
cp docker_files/Dockerfile_gpu Dockerfile
docker build -t buboqa .
```

- Run the Docker image on GPU with nvidia-docker installed. Notice that we are mounting the current directory so that data persists.
```
nvidia-docker run -it --rm \
  -v "$(pwd)":/code \
  buboqa
```
- OR ... Run the Docker image on CPU (not tested)
```
docker run -it --rm \
  -v "$(pwd)":/code \
  buboqa
```
- Exit shell when needed
```
$  exit
```
