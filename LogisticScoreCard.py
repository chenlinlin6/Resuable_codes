
# coding=utf-8
#Author: chengsong
#Time: 2020-03-15

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time
plt.rc('font', family='SimHei', size=13)
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from math import log


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S')



# 计算某个数据集的信息熵，标签为y
def cal_entr(data):
    p1=np.mean(data['y']==1)
    if p1==1 or p1==0:
        return 0
    else:
        return -p1*np.log(p1)-(1-p1)*np.log(1-p1)

# 对数据集的某一列col遍历所有数值，寻找信息增益最大的切分位点
def get_cut_not_null(data, col):
    max_info_gain=0.001
    cut_point='无最佳切分点'
    for i in np.sort(data[col].unique())[:-1]:
        data_01=data[data[col]<i]
        data_02=data[data[col]>=i]
        data_01_p=len(data_01)/len(data)
        data_02_p=len(data_02)/len(data)
        info_gain=cal_entr(data)-data_01_p*cal_entr(data_01)-data_02_p*cal_entr(data_02)

        if info_gain>max_info_gain:
            max_info_gain=info_gain
            cut_point=i
    return cut_point, max_info_gain



# 切分的深度以及切分后的信息增益是否能大于最小值0.001
deep=2
cut_list=[]
# 转变样式后的结果列表
col_cut_result=[]



def get_cut_not_null_all(data, col,deep=2):
    if deep>0:
        cut_point, max_info_gain=get_cut_not_null(data,col)
#         print('cut_point', cut_point)
        cut_list.append([col, cut_point])
        if cut_point != '无最佳切分点':
            data_01=data[data[col]>cut_point]
            data_02=data[data[col]<=cut_point]
            deep-=1
            if len(data_01)>2:
                get_cut_not_null_all(data_01, col,deep)
            if len(data_02)>2:
                get_cut_not_null_all(data_02, col,deep)
    
    
# 更改展示样式
def transform_cut_point(cut_list):
    cut_list=[[i ,j] for i, j in pd.DataFrame(cut_list,columns=['col', 'cut_point']).sort_values(by=['col', 'cut_point']).values]
    for col in pd.DataFrame(cut_list, columns=['col', 'cut_point'])['col'].unique():
        col_cut_list=[]
        for i, cut_point in cut_list:
            if i == col:
                col_cut_list.append(cut_point)
        col_cut_result.append([col,col_cut_list])
    return col_cut_result




# 切分多个连续变量
def get_all_col_cut_points(data):
    for col in data.columns:
        if col != 'y':
            data_input=data[[col,'y']]
            data_not_null=data_input.dropna(subset=[col])
            if data_input[col].isnull().sum()>0:
                cut_list.append([col,'null'])
                get_cut_not_null_all(data_input, col, 3)
            else: 
                get_cut_not_null_all(data_input, col, 3)
    transform_cut_point(cut_list)
    logging.info('连续变量切分完成')






# 计算没有离散的连续型变量的iv值
iv_list=[]
def cal_no_cat_iv(data):
    logging.info('计算没有离散化的连续型变量的iv值')
    for col in data.columns:
        if col != 'y':
            iv=0
            for i in data[col].unique():
            #     计算好和坏在这个数值下的比例，然后计算iv值
                p1=len(data[(data[col]==i)&(data['y']==1)])/len(data.query("y==1"))
                p0=len(data[(data[col]==i)&(data['y']==0)])/len(data.query("y==0"))

            #     计算当下的iv值
                if p1==0 or p0==0:
                    temp_iv=0
                else:
                    temp_iv=(p1-p0)*np.log(p1/p0)
                iv+=temp_iv
            iv_list.append([col, iv])
    
 


 # 对连续型变量进行按照最大熵离散化
def get_col_cut(data):
    data_cut_result=data.copy()
    for col,cut_lists in col_cut_result:
        cluster=1
        data_col=data[col]
        data_cut_result.loc[data_col.notnull(),col]=str(cluster)
        
        for cut_point in cut_lists:
            if cut_point != 'null' and cut_point!='无最佳切分点':
                cluster+=1
#                 print(data_col.unique())
#                 print(data_cut_result.loc[:, col].unique())
                data_cut_result.loc[data_col>cut_point, col]=str(cluster)
    logging.info('根据cut的切分离散型变量完成')
    return data_cut_result


# 对连续型变量进行离散化，然后进行范围表示
def get_col_cut_range(data):
	data_cut_range=data.copy()
	for col, cut_lists in col_cut_result:
		data_col=data_cut_range[col]
		for i in range(0,len(cut_lists)-1):
			if cut_lists[i] != 'null':
				data_cut_range.loc[(data_col.notnull())&(data_col>=cut_lists[i])&(data_col<cut_lists[i+1])]='[%f, %f]'%(cut_lists[i], cut_lists[i+1])
			else:

		if min(cut_lists) !='null':
			data_cut_range.loc[(data_col.notnull())&(data_col<min(cut_lists))]='[-, %f]'%min(cut_lists)
		if max(cut_lists) !='null':
			data_cut_range.loc[(data_col.notnull())&(data_col>=max(cut_lists))]='[%f, -]'%max(cut_lists)
	# 空值单独作为一个分类
    logging.info('根据cut的切分离散型变量范围完成')

	return data_cut_range


# 对离散化后的数据表进行计算iv值
iv_cat_list=[]
def cal_cat_iv(data):
    logging.info('计算离散化的连续型变量的iv值')
    for col in data.columns:
        if col != 'y':
            iv=0
            for i in data[col].unique():
            #     计算好和坏在这个数值下的比例，然后计算iv值
                p1=len(data[(data[col]==i)&(data['y']==1)])/len(data.query("y==1"))
                p0=len(data[(data[col]==i)&(data['y']==0)])/len(data.query("y==0"))

            #     计算当下的iv值
                if p1==0 or p0==0:
                    temp_iv=0
                else:
                    temp_iv=(p1-p0)*np.log(p1/p0)
                iv+=temp_iv
            iv_cat_list.append([col, iv])
    


# 根据离散化后的数据表进行woe化
def get_woe(data_cut_result):
    logging.info('数据woe化进行中。。。。。')
    data_woe=data_cut_result.copy()

    for col in data_cut_result.columns[data.columns != 'y']:
        data_col=data_cut_result[[col,'y']]
        total_0=len(data_col[(data_col['y'].notnull())&(data_col['y']==0)])
        total_1=len(data_col[(data_col['y'].notnull())&(data_col['y']==1)])


        for cluster in data_col[col].unique():
            cluster_0=len(data_col[(data_col[col]==cluster)&(data_col['y']==0)])
            cluster_1=len(data_col[(data_col[col]==cluster)&(data_col['y']==1)])
            p0=cluster_0/total_0
            p1=cluster_1/total_1

            woe_cluster=np.log(p0/p1)

            data_woe.loc[data_col[col]==cluster, col]=woe_cluster
    logging.info('数据woe化转换完成')
    return data_woe


# 剔除掉iv值小于0.1，相关性>0.8的变量（默认参数剔除iv值<0.1，相关系数>0.8的变量），返回最后入模的变量名列表
def get_iv_corr_col(data_woe, min_iv=0.1, max_corr=0.8):
    logging.info('根据IV值大于%s，相关系数小于%s选取变量'%(min_iv, max_corr))
#    选取高于0.1的iv值变量
    iv_df=pd.DataFrame(iv_cat_list,columns=['col', 'iv_value'])
#     筛选出iv值高于0.1的列名
    iv_cols=iv_df.query("iv_value>=0.1")['col'].tolist()
    
#     计算woe化以后各列的相关性矩阵
    data_woe_corr=data_woe[iv_cols].astype(float).corr()
    
    
    del_cols=[]
# 遍历行，找到相关系数大于0.8的数据，排除掉相同的变量名(其相关系数为1)，找到iv值较大的一个
    for col_1 in data_woe_corr.columns:
        for col_2 in data_woe_corr[data_woe_corr[col_1]>max_corr].index:
            if col_1 != col_2:
                iv_1=iv_df[iv_df['col']==col_1]['iv_value'].values
                iv_2=iv_df[iv_df['col']==col_2]['iv_value'].values
                if iv_1>iv_2:
                    del_cols.append(col_2)
                    
                    
    col_result=[col for col in iv_cols if col not in del_cols]
    logging.info("总共%s个变量，最终筛选出%s个变量"%(len(iv_df), len(col_result)))
    return col_result



# 通过iv值和相关性以及逻辑回归L1选择变量(默认参数剔除iv值<0.1，相关系数>0.8的变量，模型L1正则化选择变量)
def get_iv_corr_l1_col(data_woe, min_iv=0.1, max_corr=0.8):
    logging.info('根据IV值大于%s，相关系数小于%s选取变量'%(min_iv, max_corr))
#    选取高于0.1的iv值变量
    iv_df=pd.DataFrame(iv_cat_list,columns=['col', 'iv_value'])
#     筛选出iv值高于0.1的列名
    iv_cols=iv_df.query("iv_value>=0.1")['col'].tolist()
    
#     计算woe化以后各列的相关性矩阵
    data_woe_corr=data_woe[iv_cols].astype(float).corr()
    
    
    del_cols=[]
# 遍历行，找到相关系数大于0.8的数据，排除掉相同的变量名(其相关系数为1)，找到iv值较大的一个
    for col_1 in data_woe_corr.columns:
        for col_2 in data_woe_corr[data_woe_corr[col_1]>max_corr].index:
            if col_1 != col_2:
                iv_1=iv_df[iv_df['col']==col_1]['iv_value'].values
                iv_2=iv_df[iv_df['col']==col_2]['iv_value'].values
                if iv_1>iv_2:
                    del_cols.append(col_2)
                    
                    
    col_result=[col for col in iv_cols if col not in del_cols]
    
#     通过l1正则化筛选变量
    lr=LogisticRegression(penalty='l1',C=1.0)
    lr.fit(data_woe[col_result].astype(float),data_woe['y'])
    
    col_result=[col_result[i] for i in range(len(col_result)) if lr.coef_[0][i] !=0 ]
    
    
    
    
    logging.info("总共%s个变量，最终筛选出%s个变量"%(len(iv_df), len(col_result)))
    
    return col_result



# 根据IV值绘制直方图
# 生成范围和标签关联的表
def plot_iv_graph(data):
	range_data=data_cut_range[col_result].join(data['y'])
	# 准备好iv值列表
	iv_data=pd.DataFrame(iv_cat_list,columns=['col', 'iv_value'])

	# 计算每一列各个区间的个数，坏样本的占比
	def cal_range_value(df,col):
		result_df=df.groupby(col).agg({'y':['count', 'mean']})
		result_df.columns=result_df.columns.droplevel().rename({'count':'样本数', 'mean':'坏样本率'})
		return result_df




	for col in range_data.columns[:-1]:
		# 给定指定列和y值，返回计算后的样本数和坏样本率
		result_df=range_data[[col,'y']].apply(cal_range_value, col)
		figure,ax=plt.subplots(figsize=(8,6))


		result_df.plot(y='count',kind='bar',ax=ax)
		result_df.plot(y='mean',kind='line',ax=ax,secondary_y=True,style='-o')

		plt.title('IV值: %f'%iv_data[iv_data['col']==col]['iv_value'].values[0])
		plt.show()



# 生成模型报告, 可以调整箱的个数
def model_report(data_woe,bins=20):
	lr_model=LogisticRegression(C=1, penalty='l1')

	lr_model.fit(data_woe[col_result], 'y')
	# 预测
	y_prob=lr_model.predict_proba(data_woe[col_result]).iloc[:, 1]

	lis=pd.DataFrame({'y':data_woe['y'], 'y_prob':y_prob})
	
	# 计算总的负样本个数
	total_bad=lis['y'].sum().values[0]


	# 计算总的正样本个数
	total_good=lis.query("y==0").count().values[0]

	# 在不同划分区间下计算这几个指标，ks值，负样本个数，正样本个数，负样本累计个数，正样本累计个数，捕获率，负样本占比
	lis['bins']=pd.qcut(lis['y_prob'],q=bins)

	lis.sort_valeus(by='y_prob',ascending=False,inplace=True)

	# 开始计算
	# 先计算每个区间负样本和正样本的个数
	def cal_neg_pos_num(data, col):
		neg_num=data[col].sum().values
		pos_num=sum(data[col]==0)
		return pd.DataFrame([[neg_num, pos_num]], columns=['负样本数', '正样本数'], index=[0])

	init_result=lis.groupby('bins').apply(cal_neg_pos_num, 'y')
	init_result.sort_values(by='bins',ascending=False)
	# 计算负样本和正样本的累计个数
	init_result['负样本累计个数']=init_result['负样本数'].cumsum()
	init_result['正样本累计个数']=init_result['正样本数'].cumsum()

	init_result['捕获率/负样本累计占比']=init_result['负样本累计个数']/total_bad
	init_result['正样本累计占比']=init_result['正样本累计个数']/total_good

	init_result['负样本占比']=init_result['负样本数']/(init_result['负样本数']+init_result['正样本数'])

	init_result['ks']=init_result['捕获率/负样本累计占比']/init_result['正样本累计占比']


	return init_result



def score(data,cut_points_result,score_card):
    pred_prob=model.predict_proba(data[col_result])
    false_positive_rate, recall, thresholds = roc_curve(data['y'], pred_prob)
    roc_auc = auc(false_positive_rate, recall)
    ks = max(recall - false_positive_rate)
    result={}
    result['auc']=roc_auc
    result['ks']=ks
    return result

# 绘制分数分布
def plot_distribution(score,label):
    score_distribution=pd.DataFrame(dict(label=label,score=score))

    sns.distplot(score_distribution.query("label==1")['score'],color='r',label='bad')
    sns.distplot(score_distribution.query("label==0")['score'],color='g',label='good')
    
    plt.legend(loc='upper right')


from sklearn.metrics import roc_curve, roc_auc_score
# 绘制roc曲线
def plot_roc(y_true, y_prob, filename=False):
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



# 绘制ks曲线
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


