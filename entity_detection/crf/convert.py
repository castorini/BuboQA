from argparse import ArgumentParser




def convert(filename, output):
    fin = open(filename, 'r')
    fout = open(output, 'w')
    for line in fin.readlines():
        items = line.strip().split('\t')
        sent, label = items[5], items[6]
        for word, tag in zip(sent.strip().split(), label.strip().split()):
            fout.write("{}\t{}\n".format(word,tag))
        fout.write("\n")
    fout.close()

if __name__=='__main__':
    parser = ArgumentParser(description='Convert dataset to stanford format for training')
    parser.add_argument('--data_dir', type=str, default="../../../data/processed_simplequestions_dataset/train.txt")
    parser.add_argument('--save_path', type=str, default="data/stanford.train")
    args = parser.parse_args()
    convert(args.data_dir, args.save_path)
