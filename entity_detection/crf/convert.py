
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
    convert("../../../data/processed_simplequestions_dataset/train.txt", "data/stanford.train")
    convert("../../../data/processed_simplequestions_dataset/valid.txt", "data/stanford.valid")
    convert("../../../data/processed_simplequestions_dataset/test.txt", "data/stanford.test")
