
def get_span(label, index2tag):
    span = []
    st = 0
    en = 0
    flag = False
    for k in range(len(label)):
        if index2tag[label[k]] == 'I' and flag == False:
            flag = True
            st = k
        if index2tag[label[k]] != 'I' and flag == True:
            flag = False
            en = k
            span.append((st, en))
            st = 0
            en = 0
    if st != 0 and en == 0:
        en = k
        span.append((st, en))
    return span

def evaluation(gold, pred, index2tag):
    right = 0
    predicted = 0
    total_en = 0
    for i in range(len(gold)):
        gold_batch = gold[i]
        pred_batch = pred[i]
        for j in range(len(gold_batch)):
            gold_label = gold_batch[j]
            pred_label = pred_batch[j]
            gold_span = get_span(gold_label, index2tag)
            pred_span = get_span(pred_label, index2tag)
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
    return precision, recall, f1
