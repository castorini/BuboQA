from random import randint, uniform
from subprocess import call

epochs = 20
count = 40
for id in range(count):
    learning_rate = 10 ** uniform(-6, -2)
    d_hidden = randint(80, 400)
    n_layers = randint(2, 3)
    dropout = uniform(0.1, 0.5)
    clip = uniform(0.5, 0.7)

    command = "python train.py --dev_every 2200 --log_every 2200 --batch_size 32 " \
                "--epochs {} --lr {} --d_hidden {} --n_layers {} --dropout_prob {} --clip_gradient {} >> " \
                    "hyperparameter_results.txt".format(epochs, learning_rate, d_hidden, n_layers, dropout, clip)

    print("Running: " + command)
    call(command, shell=True)
