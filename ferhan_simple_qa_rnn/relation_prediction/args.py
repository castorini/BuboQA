import os

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Simple QA model - Ferhan Ture')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rnn_type', type=str, default='gru', help="use 'gru' or 'lstm'")
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_weight_decay', type=float, default=0.0)
    parser.add_argument('--not_bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--dev_every', type=int, default=2300)
    parser.add_argument('--save_every', type=int, default=4500)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=3, help="number of epochs to wait before early stopping")
    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use') # use -1 for CPU
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducing results')
    parser.add_argument('--save_path', type=str, default='saved_checkpoints')
    parser.add_argument('--data_cache', type=str, default=os.path.join(os.getcwd(), 'data_cache'))
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), 'vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb') # fine-tune the word embeddings
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--hits', type=int, default=3, help="number of top results to output")
    parser.add_argument('--results_path', type=str, default='results')
    args = parser.parse_args()
    return args
