def plot_distribution(score,label):
    score_distribution=pd.DataFrame(dict(label=label,score=score))

    sns.distplot(score_distribution.query("label==1")['score'],color='r',label='bad')
    sns.distplot(score_distribution.query("label==0")['score'],color='g',label='good')
    
    plt.legend(loc='upper right')
    