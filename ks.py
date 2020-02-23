from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import numpy as np
def plot_ks(y_true, y_prob, thresholds_num):
    def tpr_fpr_delta(y_true, y_prob, threshold):
        y_pred=[int(i>threshold) for i in y_prob]
        tn, fp, fn, tp=confusion_matrix(y_true, y_pred).ravel()
        fpr=fp/(fp+tn)
        tpr=tp/(fn+tp)
        delta=tpr-fpr
        return fpr, tpr, delta

    thresholds_num=1000
    thresholds=np.linspace(0,1,thresholds_num)
    fprs=[]
    tprs=[]
    deltas=[]
    for thres in thresholds:
        fpr, tpr, delta=tpr_fpr_delta(y_test, y_prob[:,1], thres)
        fprs.append(fpr)
        tprs.append(tpr)
        deltas.append(delta)

    max_delta_idx=np.argmax(np.array(deltas))
    ks_value=np.max(deltas)

    target_fpr=fprs[max_delta_idx]
    target_tpr=tprs[max_delta_idx]
    target_threshold=thresholds[max_delta_idx]

    x=[target_threshold, target_threshold]
    y=[target_fpr, target_tpr]

    plt.figure(figsize=(8,4))
    for i in range(len(x)):    
        plt.scatter(x[i], y[i],color='blue',s=20)

    plt.plot([x[0],y[0]],[x[1],y[1]], linestyle='--' )

    plt.plot(thresholds, tprs, label='TPR',color='r')
    plt.plot(thresholds, fprs, label='FPR',color='b')
    plt.plot(thresholds, deltas, label='KS',color='y')


    plt.annotate(s='TPR: {:.3f}'.format(target_tpr),xy=(target_threshold, target_tpr),xytext=(target_threshold, target_tpr+0.05), arrowprops=dict(arrowstyle='<-',color='k'), fontsize=15)
    plt.annotate(s='FPR: {:.3f}'.format(target_fpr),xy=(target_threshold, target_fpr),xytext=(target_threshold-0.1, target_fpr-0.1), arrowprops=dict(arrowstyle='<-',color='k'),fontsize=15)

    plt.annotate(s='KS: {:.3f}'.format(ks_value),xy=(target_threshold-0.1, ks_value+0.05), fontsize=15)

    plt.legend(loc='upper right') 