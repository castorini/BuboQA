
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
    convert("stanford-ner/data/stanford.predicted.valid", "../../data/processed_simplequestions_dataset/lineids_valid.txt" , "query_text/query.valid")
    convert("stanford-ner/data/stanford.predicted.test", "../../data/processed_simplequestions_dataset/lineids_test.txt" , "query_text/query.test")