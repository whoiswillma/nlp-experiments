def clean_preds(preds):
    prev_label = "O"
    for i in range(len(preds)):
        if preds[i].startswith("O"):
            prev_label = "O"
        else:
            curr_label = preds[i][2:]
            if preds[i].startswith("I") and curr_label != prev_label:
                preds[i] = "B-" + curr_label
            prev_label = curr_label
    return preds


def calc_scores(true_values, pred_values, id2label):
    true_labels = [id2label[i] for i in true_values.tolist()]
    pred_labels = [id2label[i] for i in pred_values.tolist()]
    pred_labels = clean_preds(pred_labels)

    TP, FP, FN = 0, 0, 0
    match = False

    for true, pred in zip(true_labels, pred_labels):
        if match:
            if true.startswith("I"):
                if true != pred:
                    FP += 1
                    FN += 1
                    match = False
            elif pred.startswith("I"):
                FP += 1
                FN += 1
                match = False
            else:
                TP += 1
                match = False

        if true.startswith("B"):
            if true != pred:
                FN += 1
        if pred.startswith("B"):
            if true != pred:
                FP += 1
            else:
                match = True

    return TP, FP, FN


def get_scores(TP, FP, FN):
    precision, recall, f1 = 0, 0, 0

    if (TP + FP != 0):
        precision = TP / (TP+FP)
    if (TP + FN != 0):
        recall = TP / (TP + FN)
    if (precision + recall != 0):
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def to_label_list(true_values, pred_values):
    true_labels = [id2label[i] for i in true_values.tolist()]
    pred_labels = [id2label[i] for i in pred_values.tolist()]
    pred_labels = clean_preds(pred_labels)
    return true_labels, pred_labels
