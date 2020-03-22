import numpy as np
import pandas as pd



def get_arr_bin(arr_base, arr_new, bins=10):
    arr = np.r_[arr_base, arr_new]
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    arr_bins = np.linspace(arr_min-0.001, arr_max+0.001, bins+1)
    # print(arr_bins)
    # 返回单个值value对应的组别编号，其中编号从0开始计数，若有10组，则编号为0~9
    def get_bin(value):
        delta = [1 if(value-i)>0 else 0 for i in arr_bins]
        group = np.argwhere(np.diff(delta) != 0)[0][0]
        return 'group_' + str(group)
    
    base_bins = [get_bin(i) for i in arr_base]
    base_bins_counts = pd.Series(base_bins).value_counts().sort_index().reset_index()
    base_bins_counts.columns = ['group', 'count']
    base_bins_counts['ratio'] = base_bins_counts['count'] / sum(base_bins_counts['count'])
    base_bins_counts.drop(['count'], axis=1, inplace=True)
    
    
    new_bins = [get_bin(i) for i in arr_new]
    new_bins_counts = pd.Series(new_bins).value_counts().sort_index().reset_index()
    new_bins_counts.columns = ['group', 'count']
    new_bins_counts['ratio'] = new_bins_counts['count'] / sum(new_bins_counts['count'])
    new_bins_counts.drop(['count'], axis=1, inplace=True)
    
    base_zeros = np.zeros(bins)
    indices = [int(i[-1]) for i in base_bins_counts['group']]
    for i, j in enumerate(indices):
        base_zeros[j] = base_bins_counts['ratio'].values[i]
    
    new_zeros = np.zeros(bins)
    indices = [int(i[-1]) for i in new_bins_counts['group']]
    for i, j in enumerate(indices):
        new_zeros[j] = new_bins_counts['ratio'].values[i]
    
    def ff(i, j):
        delta = i - j
        i = 0.00001 if i == 0 else i
        j = 0.00001 if j == 0 else j
        lg = np.log(i / j)
        return delta * lg
    
    #print(base_zeros)
    #print(new_zeros)
    return sum([ff(i, j) for i,j in zip(base_zeros, new_zeros)])