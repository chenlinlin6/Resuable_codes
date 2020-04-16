def get_ks(y_true, y_prob, thresholds_num=250):
    # 生成一系列阈值
    thresholds = np.linspace(np.min(y_prob), np.max(y_prob), thresholds_num) 
    
    def tpr_fpr_delta(threshold):
        y_pred = np.array([int(i>threshold) for i in y_prob])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp+tn)
        tpr = tp / (tp+fn)
        delta = tpr - fpr
        return delta

    deltas = np.vectorize(tpr_fpr_delta)(thresholds)
    max_delta = np.max(deltas)
    return max_delta


def evaluate_model(y_true, y_pred, y_prob):
    assert len(y_true) == len(y_pred)
    assert len(y_true) == len(y_prob)
    
    acc = accuracy_score(y_true, y_pred)
    ks = get_ks(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    decription = 'AUC:{:.3f}, KS:{:.3f}, F1:{:.3f}, ACC:{:.3F}, Recall:{:.3f}, Precision:{:.3f}'
    print(decription.format(auc, ks, f1, acc, recall, precision))