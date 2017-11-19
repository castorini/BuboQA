from argparse import ArgumentParser


def convert(fileName, idFile, outputFile):
    fin = open(fileName)
    fid = open(idFile)
    fout = open(outputFile, "w")
    word_list = []
    pred_query = []
    line_id = []
    for line in fid.readlines():
        line_id.append(line.strip())
    index = 0
    for line in fin.readlines():
        if line == '\n':
            if len(pred_query) == 0:
                pred_query = word_list
            fout.write("{} %%%% {}\n".format(line_id[index], " ".join(pred_query)))
            index += 1
            pred_query = []
            word_list = []
        else:
            word, gold_label, pred_label = line.strip().split()
            word_list.append(word)
            if pred_label == 'I':
                pred_query.append(word)
    if (index != len(line_id)):
        print("Length Error")

if __name__=="__main__":
    parser = ArgumentParser(description='Convert result to query text')
    parser.add_argument('--data_dir', type=str, default="stanford-ner/data/stanford.predicted.valid")
    parser.add_argument('--valid_line', type=str, default="../../data/processed_simplequestions_dataset/lineids_valid.txt")
    parser.add_argument('--results_path', type=str, default="query_text/query.valid")
    args = parser.parse_args()
    convert(args.data_dir, args.valid_line, args.results_path)
