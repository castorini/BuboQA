import sys

def get_span(label):
    span = []
    st = -1
    en = -1
    flag = False

    for k, item in enumerate(label):
        if item == 'I' and flag == False:
            flag = True
            st = k
        if item != 'I' and flag == True:
            flag = False
            en = k
            span.append((st, en))
            st = -1
            en = -1
    if st != -1 and en == -1:
        en = len(label)
        span.append((st, en))
    return span

def evaluation(filename):
    fin = open(filename, 'r')
    pred = []
    gold = []
    right = 0
    predicted = 0
    total_en = 0
    for line in fin.readlines():
        if line == '\n':
            gold_span = get_span(gold)
            pred_span = get_span(pred)
            total_en += len(gold_span)
            predicted += len(pred_span)
            for item in pred_span:
                if item in gold_span:
                    right += 1
            gold = []
            pred = []
        else:
            word, gold_label, pred_label = line.strip().split()
            gold.append(gold_label)
            pred.append(pred_label)

    if gold != [] or pred != []:
        gold_span = get_span(gold)
        pred_span = get_span(pred)
        total_en += len(gold_span)
        predicted += len(pred_span)
        for item in pred_span:
            if item in gold_span:
                right += 1

    if predicted == 0:
        precision = 0
    else:
        precision = right / predicted
    if total_en == 0:
        recall = 0
    else:
        recall = right / total_en
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    print("Precision",precision,"Recall",recall,"F1",f1,"right",right,"predicted",predicted,"total",total_en)


if __name__=='__main__':
    if len(sys.argv) != 2:
        print("Need to specify the file")
    filename = sys.argv[1]
    evaluation(filename)
