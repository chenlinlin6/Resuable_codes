# 定义计算psi的函数
def psi_for_continue_var(expected_array, actual_array, bins=10, bucket_type='bins', detail=False, save_file_path=None):
    '''
    ----------------------------------------------------------------------
    功能: 计算连续型变量的群体性稳定性指标（population stability index ,PSI）
    ----------------------------------------------------------------------
    :param expected_array: numpy array of original values，基准组
    :param actual_array: numpy array of new values, same size as expected，比较组
    :param bins: number of percentile ranges to bucket the values into，分箱数, 默认为10
    :param bucket_type: string, 分箱模式，'bins'为等距均分，'quantiles'为按等频分箱
    :param detail: bool, 取值为True时输出psi计算的完整表格, 否则只输出最终的psi值
    :param save_file_path: string, csv文件保存路径. 默认值=None. 只有当detail=Ture时才生效.
    ----------------------------------------------------------------------
    :return psi_value: 
            当detail=False时, 类型float, 输出最终psi计算值;
            当detail=True时, 类型pd.DataFrame, 输出psi计算的完整表格。最终psi计算值 = list(psi_value['psi'])[-1]
    ----------------------------------------------------------------------
    示例：
    >>> psi_for_continue_var(expected_array=df['score'][:400],
                             actual_array=df['score'][401:], 
                             bins=5, bucket_type='bins', detail=0)
    >>> 0.0059132756739701245
    ------------
    >>> psi_for_continue_var(expected_array=df['score'][:400],
                             actual_array=df['score'][401:], 
                             bins=5, bucket_type='bins', detail=1)
    >>>
    	score_range	expecteds	expected(%)	actucalsactucal(%)ac - ex(%)ln(ac/ex)psi	max
    0	[0.021,0.2095]	120.0	30.00	152.0	31.02	1.02	0.033434	0.000341	
    1	(0.2095,0.398]	117.0	29.25	140.0	28.57	-0.68	-0.023522	0.000159	
    2	(0.398,0.5865]	81.0	20.25	94.0	19.18	-1.07	-0.054284	0.000577	<<<<<<<
    3	(0.5865,0.7751]	44.0	11.00	55.0	11.22	0.22	0.019801	0.000045	
    4	(0.7751,0.9636]	38.0	9.50	48.0	9.80	0.30	0.031087	0.000091	
    5	>>> summary	400.0	100.00	489.0	100.00	NaN	NaN	0.001214	<<< result
    ----------------------------------------------------------------------
    知识:
    公式： psi = sum(（实际占比-预期占比）* ln(实际占比/预期占比))
    一般认为psi小于0.1时候变量稳定性很高，0.1-0.25一般，大于0.25变量稳定性差，建议重做。
    相对于变量分布(EDD)而言, psi是一个宏观指标, 无法解释两个分布不一致的原因。但可以通过观察每个分箱的sub_psi来判断。
    ----------------------------------------------------------------------
    '''
    import math
    import numpy as np
    import pandas as pd
    
    expected_array = pd.Series(expected_array).dropna()
    actual_array = pd.Series(actual_array).dropna()
    if isinstance(list(expected_array)[0], str) or isinstance(list(actual_array)[0], str):
        raise Exception("输入数据expected_array只能是数值型, 不能为string类型")
        
    """step1: 确定分箱间隔"""
    def scale_range(input_array, scaled_min, scaled_max):
        '''
        ----------------------------------------------------------------------
        功能: 对input_array线性放缩至[scaled_min, scaled_max]
        ----------------------------------------------------------------------
        :param input_array: numpy array of original values, 需放缩的原始数列
        :param scaled_min: float, 放缩后的最小值
        :param scaled_min: float, 放缩后的最大值
        ----------------------------------------------------------------------
        :return input_array: numpy array of original values, 放缩后的数列
        ----------------------------------------------------------------------
        '''
        input_array += -np.min(input_array) # 此时最小值放缩到0
        if scaled_max == scaled_min:
            raise Exception('放缩后的数列scaled_min = scaled_min, 值为{}, 请检查expected_array数值！'.format(scaled_max))
        scaled_slope = np.max(input_array) * 1.0 / (scaled_max - scaled_min)
        input_array /= scaled_slope
        input_array += scaled_min
        return input_array
    
    # 异常处理，所有取值都相同时, 说明该变量是常量, 返回999999
    if np.min(expected_array) == np.max(expected_array):
        return 999999
    
    breakpoints = np.arange(0, bins + 1) / (bins) * 100 # 等距分箱百分比
    if 'bins' == bucket_type:        # 等距分箱
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif 'quantiles' == bucket_type: # 等频分箱
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])
 
    """step2: 统计区间内样本占比"""
    def generate_counts(arr, breakpoints):
        '''
        ----------------------------------------------------------------------
        功能: Generates counts for each bucket by using the bucket values 
        ----------------------------------------------------------------------
        :param arr: ndarray of actual values
        :param breakpoints: list of bucket values
        ----------------------------------------------------------------------
        :return cnt_array: counts for elements in each bucket, length of breakpoints array minus one
        :return score_range_array: 分箱区间
        ----------------------------------------------------------------------
        '''
        def count_in_range(arr, low, high, start):
            '''
            ----------------------------------------------------------------------
            功能: 统计给定区间内的样本数(Counts elements in array between low and high values)
            ----------------------------------------------------------------------
            :param arr: ndarray of actual values
            :param low: float, 左边界
            :param high: float, 右边界
            :param start: bool, 取值为Ture时，区间闭合方式[low, high],否则为(low, high]
            ----------------------------------------------------------------------
            :return cnt_in_range: int, 给定区间内的样本数
            ----------------------------------------------------------------------
            '''
            if start:
                cnt_in_range = len(np.where(np.logical_and(arr >= low, arr <= high))[0])
            else:
                cnt_in_range = len(np.where(np.logical_and(arr > low, arr <= high))[0])
            return cnt_in_range
 
        cnt_array = np.zeros(len(breakpoints) - 1)
        score_range_array = [''] * (len(breakpoints) - 1)
        for i in range(1, len(breakpoints)):
            cnt_array[i-1] = count_in_range(arr, breakpoints[i-1], breakpoints[i], i==1)
            if 1 == i:
                score_range_array[i-1] = '[' + str(round(breakpoints[i-1], 4)) + ',' + str(round(breakpoints[i], 4)) + ']'
            else:   
                score_range_array[i-1] = '(' + str(round(breakpoints[i-1], 4)) + ',' + str(round(breakpoints[i], 4)) + ']'
                                                                                
        return (cnt_array, score_range_array)
 
    expected_cnt = generate_counts(expected_array, breakpoints)[0]
    expected_percents = expected_cnt / len(expected_array)
    actual_cnt = generate_counts(actual_array, breakpoints)[0]
    actual_percents = actual_cnt / len(actual_array)
    delta_percents = actual_percents - expected_percents
    score_range_array = generate_counts(expected_array, breakpoints)[1]
                                                                               
    """step3: 区间放缩"""
    def sub_psi(e_perc, a_perc):
        '''
        ----------------------------------------------------------------------
        功能: 计算单个分箱内的psi值。Calculate the actual PSI value from comparing the values.
             Update the actual value to a very small number if equal to zero
        ----------------------------------------------------------------------
        :param e_perc: float, 期望占比
        :param a_perc: float, 实际占比
        ----------------------------------------------------------------------
        :return value: float, 单个分箱内的psi值
        ----------------------------------------------------------------------
        '''
        if a_perc == 0: # 实际占比
            a_perc = 0.001
        if e_perc == 0: # 期望占比
            e_perc = 0.001
        value = (e_perc - a_perc) * np.log(e_perc * 1.0 / a_perc)
        return value
    
    """step4: 得到最终稳定性指标"""
    sub_psi_array = [sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))]
    if detail:
        psi_value = pd.DataFrame()
        psi_value['score_range'] = score_range_array
        psi_value['expecteds'] = expected_cnt
        psi_value['expected(%)'] = expected_percents * 100
        psi_value['actucals'] = actual_cnt
        psi_value['actucal(%)'] = actual_percents * 100
        psi_value['ac - ex(%)'] = delta_percents * 100
        psi_value['actucal(%)'] = psi_value['actucal(%)'].apply(lambda x: round(x, 2))
        psi_value['ac - ex(%)'] = psi_value['ac - ex(%)'].apply(lambda x: round(x, 2))
        psi_value['ln(ac/ex)'] = psi_value.apply(lambda row: np.log((row['actucal(%)'] + 0.001)                                                                   / (row['expected(%)'] + 0.001)), axis=1)
        psi_value['psi'] = sub_psi_array
        flag = lambda x: '<<<<<<<' if x == psi_value.psi.max() else ''
        psi_value['max'] = psi_value.psi.apply(flag)
        psi_value = psi_value.append([{'score_range':'>>> summary', 
                                       'expecteds': sum(expected_cnt),
                                       'expected(%)':100, 
                                       'actucals': sum(actual_cnt),
                                       'actucal(%)':100, 
                                       'ac - ex(%)': np.nan,
                                       'ln(ac/ex)': np.nan,
                                       'psi': np.sum(sub_psi_array),
                                       'max':'<<< result'}], ignore_index=True)
        if save_file_path:
            if not isinstance(save_file_path, str):
                raise Exception('参数save_file_path类型必须是str, 同时注意csv文件后缀!')
            elif not save_file_path.endswith('.csv'):
                raise Exception('参数save_file_path不是csv文件后缀，请检查!')
            psi_value.to_csv(save_file_path, encoding='utf-8', index=1)
    else:
        psi_value = np.sum(sub_psi_array)
 
    return psi_value