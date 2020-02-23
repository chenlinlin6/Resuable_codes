from sklearn.metrics import roc_curve, roc_auc_score
def plot_auc(y_true, y_prob, filename=False):
    fpr, tpr,threshold=roc_curve(y_true, y_prob)
    
    plt.rcParams['font.family']=['sans-serif']
    # plt.rcParams['font.weight']=['blod']
    plt.figure(figsize=(8,4))
    plt.plot(fpr, tpr,label='ROC')

    plt.xlabel('FPR',fontsize=15)
    plt.ylabel('Recall',fontsize=15)
    plt.title('ROC曲线',fontproperties=font,fontsize=20)
    plt.xticks(fontsize=10)
    plt.grid(False)
    plt.fill_between(fpr,tpr,0,color='#AED1E5',alpha=0.25)
    plt.annotate(s='auc '+str(auc_score),xy=(0.6,0.6),fontsize=20)
    plt.legend(loc='upper right')
    
    if filename:
        plt.savefig(filename)