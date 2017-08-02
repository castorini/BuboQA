from random import randint, uniform
from subprocess import call

epochs = 30
count = 10
for id in range(count):
    learning_rate = 1e-4
    d_hidden = randint(100, 300)
    n_layers = 2
    dropout = 0.3
    clip = 0.6

    command = "python train.py --dev_every 2200 --log_every 2200 --save_every 5000 --batch_size 32 " \
                "--epochs {} --lr {} --d_hidden {} --n_layers {} --dropout_prob {} --clip_gradient {} >> " \
                    "hyperparameter_results.txt".format(epochs, learning_rate, d_hidden, n_layers, dropout, clip)

    print("Running: " + command)
    call(command, shell=True)
