#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if_parallel = False

import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import Counter

from loss_function_dlda import lda_loss
from keras.layers import Dense, BatchNormalization, Activation
from keras.models import Sequential,load_model
from keras.regularizers import l2
from keras import callbacks
from keras.utils import multi_gpu_model,Sequence

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation,KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold

from hyperopt import hp, fmin, tpe, Trials
from sklearn.neural_network import MLPRegressor
from sklearn import svm
import joblib

# define the date generator: limit the sample size for training to 100
# 定义数据生成器：每次每类采100个样进行训练
class Datagenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        '''

        :param x_set: 训练数据
        :param y_set: 训练数据标签
        :param batch_size: 每次产生的用于训练的数据量，如 各类数据量100*类别数47
        '''
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

        self.n_splits = int(np.ceil(len(self.x) / float(self.batch_size)))
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        self.train_index = []
        self.test_index = []
        for i, j in skf.split(self.x, self.y):
            self.train_index.append(i)
            self.test_index.append(j)

    def __len__(self):
        """
           返回生成器的长度，也就是总共分批生成数据的次数。

        """
        return self.n_splits

    def __getitem__(self, num_select):
        """
           该函数返回每次我们需要的经过处理的数据。
        """

        x_test = self.x[self.test_index[num_select]]
        y_test = self.y[self.test_index[num_select]]

        return x_test, y_test

# mark sample for visulization (plot): single sample
# 划分可视化
def mark_view(xi,hour_=24):
    '''

    :param xi: 单个样本数据
    :param hour_: 样本数据的长度
    :return:
    '''
    if hour_ == 24:
        _hour = [i for i in range(1, 25)]
    elif hour_ == 48:
        _hour = [i*0.5 for i in range(1,49)]
    else:
        _hour = [i for i in range(1,hour_+1)]

    m = 20#max(xi)
    if m < 16:
        m = 16
    if hour_ == 24:
        plt.plot([1,1], [0, m], color='gray', linestyle='--')
        plt.plot([7.5,7.5],[0,m],color='gray',linestyle='--')
        plt.plot([10.5,10.5], [0, m], color='gray', linestyle='--')
        plt.plot([13.5,13.5], [0, m], color='gray', linestyle='--')
        plt.plot([16.5,16.5], [0, m], color='gray', linestyle='--')
        plt.plot([19.5,19.5], [0, m], color='gray', linestyle='--')
        plt.plot([24,24], [0, m], color='gray', linestyle='--')

    '''
    median = np.median(xi)
    m_low = median - 1
    m_high = median + 1
    plt.plot([1, 24], [median,median], color='black', linestyle='--')
    plt.plot([1, 24], [m_low,m_low], color='black', linestyle='--')
    plt.plot([1, 24], [m_high,m_high], color='black', linestyle='--')
    '''
    aver = np.mean(xi)
    m_low = aver - 2  # 2
    m_high = aver + 2 # 6
    plt.plot([1, max(_hour)], [aver,aver], color='brown', linestyle='--')
    plt.plot([1, max(_hour)], [m_low,m_low], color='r', linestyle='--')
    plt.plot([1, max(_hour)], [m_high,m_high], color='r', linestyle='--')

    m_1 = m_high + 4 # 10
    m_2 = m_1 + 4     # 14
    plt.plot([1, max(_hour)], [m_1, m_1], color='r', linestyle='--')
    plt.plot([1, max(_hour)], [m_2, m_2], color='r', linestyle='--')

    #plt.plot([1, max(_hour)], [m_1+1, m_1+1], color='pink', linestyle='--')
    #plt.plot([1, max(_hour)], [m_2+1, m_2+1], color='pink', linestyle='--')
    '''
    plt.text(2.5, 0, 'A-O:1',color='brown')
    plt.text(6.5, 0, 'M&A-M:2',color='brown')
    plt.text(11, 0, 'M-N:3',color='brown')
    plt.text(14, 0, 'A-A:4',color='brown')
    plt.text(17, 0, 'M-E:5',color='brown')
    plt.text(20.5, 0, 'A-E:6',color='brown')
    '''
    if hour_ == 24:
        plt.text(4, 0, '1',color='brown')
        plt.text(8.9, 0, '2',color='brown')
        plt.text(11.8, 0, '3',color='brown')
        plt.text(14.8, 0, '4',color='brown')
        plt.text(17.8, 0, '5',color='brown')
        plt.text(21.5, 0, '6',color='brown')

    plt.text(0, aver+0.1, 'average', color='brown')
    plt.text(0, m_low+0.1, 'lowerbound', color='pink')
    plt.text(0, m_high+0.1, 'upperbound', color='pink')
    #plt.text(0, m_low - 1, 'level-0', color='black')
    plt.text(0, m_1 - 2, 'level-1', color='black')
    plt.text(0, m_2 - 2, 'level-2', color='black')
    plt.text(0, m_2 + 1, 'level-3', color='black')

    plt.plot(_hour,xi)#,'r')
    plt.xlabel('hour(h)')
    plt.ylabel('percent(%)')
    if hour_ == 24:
        plt.xticks([1,7.5,10.5,13.5,16.5,19.5,24])
    #plt.show()

# mark sample for visulization (bar): single sample
# 柱状图-分段平均值
def mark_view_bar(xi, hour_=24, color_='b',
             if_plot_curve=True, if_plot_bar=True, if_aver_time=True):
    if hour_ == 24:
        _hour = [i for i in range(1, 25)]
    elif hour_ == 48:
        _hour = [i * 0.5 for i in range(1, 49)]
    else:
        _hour = [i for i in range(1, hour_ + 1)]

    # 各段时间的值取其平均值
    if if_aver_time == True:
        _xi = aver_time(xi)
    else:
        _xi = xi

    m = max(_xi)
    if m < 12:
        m = 12
    if hour_ == 24:
        plt.plot([0.5, 0.5], [0, m], color='gray', linestyle='--')
        plt.plot([3.5, 3.5], [0, m], color='gray', linestyle='--')
        plt.plot([7.5, 7.5], [0, m], color='gray', linestyle='--')
        plt.plot([10.5, 10.5], [0, m], color='gray', linestyle='--')
        plt.plot([13.5, 13.5], [0, m], color='gray', linestyle='--')
        plt.plot([16.5, 16.5], [0, m], color='gray', linestyle='--')
        plt.plot([19.5, 19.5], [0, m], color='gray', linestyle='--')
        plt.plot([22.5, 22.5], [0, m], color='gray', linestyle='--')
        plt.plot([24.5, 24.5], [0, m], color='gray', linestyle='--')

    aver = np.mean(xi)
    m_low = aver - 2  # 2
    # m_high = aver + 1 # 6
    plt.plot([1, max(_hour)], [aver, aver], color='brown', linestyle='--')
    plt.plot([1, max(_hour)], [m_low, m_low], color='r', linestyle='--')
    # plt.plot([1, max(_hour)], [m_high,m_high], color='r', linestyle='--')

    m_1 = aver + 2  # 10
    m_2 = m_1 + 2  # 14
    plt.plot([1, max(_hour)], [m_1, m_1], color='r', linestyle='--')
    # plt.plot([1, max(_hour)], [m_2, m_2], color='r', linestyle='--')

    if hour_ == 24:
        plt.text(2, 0, '1', color='brown')  # 3
        plt.text(5, 0, '2', color='brown')  # 4
        plt.text(8.9, 0, '3', color='brown')  # 3
        plt.text(11.8, 0, '4', color='brown')  # 3
        plt.text(14.8, 0, '5', color='brown')  # 3
        plt.text(17.8, 0, '6', color='brown')  # 3
        plt.text(20.5, 0, '7', color='brown')  # 3
        plt.text(23, 0, '8', color='brown')  # 2

    plt.text(0, aver + 0.1, 'average', color='brown')
    plt.text(0, m_low + 0.1, 'lowerbound', color='pink')
    # plt.text(0, m_high+0.1, 'upperbound', color='pink')
    plt.text(0, m_low - 1, 'level-0', color='black')
    plt.text(0, m_1 - 1, 'level-1', color='black')
    plt.text(0, m_2 - 1, 'level-2', color='black')
    # plt.text(0, m_2 + 1, 'level-3', color='black')

    plt.xlabel('hour(h)')
    plt.ylabel('percent(%)')
    if if_plot_bar == True:
        plt.bar(_hour, _xi, color=color_)

    if if_plot_curve == True:
        plt.plot(_hour, xi)
    if hour_ == 24:
        plt.xticks([1, 3, 7, 10, 13, 16, 19, 22, 24])
    plt.show()

# mark sample for visulization: multi-classes sample
# 显示分类图
def mark_view_all(x_k_, hour_=24, if_bar=False,
             if_plot_curve=True, if_plot_bar=True,if_aver_time=True):
    '''

    :param x_k_: 二维数组，聚类数据分选结果，x_k[i][j]:第i类第j个样本
    :param hour_: 样本数据的长度
    :return:
    '''
    if if_bar == True:
        for i in range(len(x_k_)):
            plt.figure()
            for j in range(len(x_k_[i])):
                mark_view_bar(x_k_[i][j], hour_=hour_, if_aver_time=if_aver_time,
                         if_plot_curve=if_plot_curve, if_plot_bar=if_plot_bar)

    else:
        for i in range(len(x_k_)):
            plt.figure()
            for j in range(len(x_k_[i])):
                mark_view(x_k_[i][j], hour_=hour_)
            plt.show()

# select samples according to its label
# 根据聚类结果对x进行分类
def select(x_,pred_y, if_first_select = True,if_print=False):
    '''

    :param x_: 聚类数据，
               若if_first_select=True，则为原始聚类数据（一维）；
               若if_first_select=False，则为上一轮聚类分拣好的数据（x_k）（二维）
    :param pred_y: x_数据标签，如AffinityPropagation.label_
    :param if_first_select: boolean,是否进行一次聚类结果的分选
    :param if_print: boolean,是否打印分选信息
    :return: x_k：二维list，聚类数据x_分选结果，x_k[i][j]:第i类第j个样本
            _k_list：一维list，与x_k对应顺序的类标号，_k_list[i]:第i类标号
            num：一维list，各类所含样本数量，num[i]:第i类样本的数量
    '''
    # 一次聚类结果分拣
    if if_first_select == True:
        _k_list = np.unique(pred_y).tolist()

        # 将数据根据分类结果进行分拣
        x_k = [[] for _ in range(len(_k_list))]

        for i in range(len(pred_y)):
            index = _k_list.index(pred_y[i])
            x_k[index].append(x_[i])
        num = [len(x_k[i]) for i in range(len(x_k))]

        if if_print == True:
            print('分类结果', pred_y)
            print('分类数',len(num))
            print('各类数量', num)

    # 二次及以上聚类结果分拣
    # x_:根据上一轮聚类分拣好的数据 x_ = select(pred_1st, x_ori, if_first_select = True)[0]
    else:
        _k_list = np.unique(pred_y).tolist()

        # 将数据根据分类结果进行分拣
        x_k_ = [[] for _ in range(len(_k_list))]

        for i in range(len(pred_y)):
            index = _k_list.index(pred_y[i])
            x_k_[index].append(x_[i])
        x_k = [[x_k_[i][j][l] for j in range(len(x_k_[i]))
                for l in range(len(x_k_[i][j]))] for i in range(len(x_k_))]
        num = [len(x_k[i]) for i in range(len(x_k))]

        if if_print == True:
            print('分类结果', pred_y)
            print('分类数',len(num))
            print('各类数量', num)

    return x_k, _k_list, num

# transfer the string label to number label
# 训练集处理：字符标签换成数字
def label_string2number(y_):
    '''

    :param y_: 一维数组，样本标签数据
    :return: y_new: 一维数组，（转换后的）样本标签数据
    '''
    if type(y_[0]) == np.str_ or type(y_[0]) == np.str: # np.array or list
        y_num_dict = Counter(y_)
        y_keys = list(y_num_dict.keys())
        y_label_dict = {y_keys[i]: i for i in range(len(y_keys))}

        y_label_number = []
        for i in range(len(y_)):
            y_label_number.append(y_label_dict[y_[i]])

        y_new = copy.deepcopy(y_label_number)
    else:
        y_new = copy.deepcopy(y_)

    return y_new

# clustering using AP
# AP聚类
def ap_cluster(x_,if_set_perf=False,percent=50,perf_value=None):
    '''

    :param x_: 一维数组，聚类数据
    :param if_set_perf=False: boolean，是否自定义AP算法的参考度（perference）
    :param percent: float(0,100)，数据点间欧氏距离的百分位数
    :param perf_value: float，AP算法的参考度，其值为相似度（距离的相反数）
    :return: ap.labels_：一维数组，聚类结果数据标签
             ap.cluster_centers_：一维数组，聚类结果类中心
             ap.cluster_centers_indices_：一维数组，聚类结果类中心在数组中的索引号
             _preference: float，AP算法执行所用的参考度
    '''

    # 得到数据点间的欧氏距离，设定AP算法的参考度（preference，值大则聚类数多）（取欧氏距离的相反数）
    if if_set_perf == True:
        if perf_value == None:
            x_dis = pairwise_distances(x_, metric='euclidean')
            _preference = - np.percentile(x_dis, percent) ** 2
        else:
            _preference = perf_value

        ap = AffinityPropagation(affinity='euclidean', max_iter=300, convergence_iter=30,
                                 damping=0.5,preference=_preference).fit(x_)
    else:
        ap = AffinityPropagation(affinity='euclidean', max_iter=300, convergence_iter=30,
                                 damping=0.5).fit(x_)
        _x_dis = ap.affinity_matrix_
        # 聚类使用的参考度
        _preference = np.median(_x_dis)

    return ap.labels_, ap.cluster_centers_, ap.cluster_centers_indices_, _preference

# evaluate the clustering results: with/without true label
# 聚类结果指标评价
def eval_index(x_ori_,x_cal_,y_true,y_pred,if_compare_train = True):
    '''

    :param x_cal_: 一维数组，用于聚类算法的数据（可能经过变换）
    :param x_ori_: 一维数组，用于聚类的数据对应的原始数据
    :param y_true: 一维数组，聚类数据x_的真实类标签
    :param y_pred: 一维数组，聚类数据x_的聚类结果类标签
    :return: [score_cal,score_ori,score_check]: 评价指标结果
    '''

    # 同质性检验
    if if_compare_train == True:
        v = metrics.v_measure_score(y_true, y_pred)
        #print('同质性与完整性的调和平均', v)
        ars = metrics.adjusted_rand_score(y_true,y_pred)
        print('调整兰德系数',ars)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
        ami = metrics.adjusted_mutual_info_score(y_true,y_pred)
        print('调整互信息', ami)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
        score_check = [v,ars,ami]
    else:
        score_check = []

    if len(np.unique(y_pred)) == 1:
        print('全部聚成一个类')
        score_cal = []
        score_ori = []
    elif len(np.unique(y_pred)) < len(y_pred):
        print('（投影）聚类数据聚类指标')
        eval_sc_cal = metrics.silhouette_score(x_cal_, y_pred, metric='euclidean')
        eval_ch_cal = metrics.calinski_harabasz_score(x_cal_, y_pred)
        eval_db_cal = metrics.davies_bouldin_score(x_cal_, y_pred)
        print('ap-x_cal', 'ch', eval_ch_cal, 'sc', eval_sc_cal, 'db', eval_db_cal)
        score_cal = [eval_ch_cal,  eval_sc_cal, eval_db_cal]

        print('原始数据聚类指标')
        eval_sc_ori = metrics.silhouette_score(x_ori_, y_pred, metric='euclidean')
        eval_ch_ori = metrics.calinski_harabasz_score(x_ori_, y_pred)
        eval_db_ori = metrics.davies_bouldin_score(x_ori_, y_pred)
        print('ap-x', 'ch', eval_ch_ori, 'sc', eval_sc_ori, 'db', eval_db_ori)
        score_ori = [eval_ch_ori, eval_sc_ori, eval_db_ori]
    else:
        print('每个样本单独成为一个类')
        score_cal = []
        score_ori = []

    return [score_cal,score_ori,score_check]

# build the objective function for optimizing hyperparameters
# 构建优化目标函数
def opt_ob(arg):
    global x_train,y_train

    units = arg['units']
    layer_num = arg['layer_num']
    reg_par = arg['reg_par']
    outdim = arg['outdim']
    opti_eigenvalue_num = arg['opti_eigenvalue_num']
    margin = arg['margin']
    epochs = arg['epochs']

    opti_eigenvalue_num = min(outdim,num_class-1,opti_eigenvalue_num)

    # 划分验证集
    n_fold = 3
    skf_t = StratifiedKFold(n_splits=n_fold, shuffle=False)

    scores = 0
    cv = 0
    epochs_actual = 0
    for train_index, valid_index in skf_t.split(x_train, y_train):
        xt, yt = x_train[train_index], y_train[train_index]
        xv, yv = x_train[valid_index], y_train[valid_index]
        '''
        sss = StratifiedKFold(n_splits=10)
        for ii,jj in sss.split(xv,yv):
            xvi,yvi = xv[jj],yv[jj]
            break
        '''
        #print(xt.shape)
        # 构建网络
        if_type = 'keras'
        if if_type == 'keras':
            x_dim = xt.shape[1]
            validation_data = None#(xvi,yvi)
            datage = Datagenerator(xt, yt, 50*156)

            earlystop = callbacks.EarlyStopping(monitor='loss',
                                                min_delta=1, patience=100)

            model = Sequential()

            if_batchnormal = True
            if if_batchnormal == True:

                model.add(Dense(units, input_shape=(x_dim,),
                                kernel_regularizer=l2(reg_par)))
                model.add(BatchNormalization())
                model.add(Activation('relu'))
                if layer_num >= 2:
                    for i in range(layer_num - 2):
                        model.add(Dense(units, kernel_regularizer=l2(reg_par)))
                        model.add(BatchNormalization())
                        model.add(Activation('relu'))

                model.add(Dense(outdim, activation='sigmoid', kernel_regularizer=l2(reg_par)))
                model.add(BatchNormalization())
            else:
                model.add(Dense(units, input_shape=(x_dim,), activation='relu',
                                kernel_regularizer=l2(reg_par)))

                if layer_num >= 2:
                    for i in range(layer_num - 2):
                        model.add(Dense(units, activation='relu', kernel_regularizer=l2(reg_par)))

                model.add(Dense(outdim, activation='linear', kernel_regularizer=l2(reg_par)))

            # model.summary()

            if if_parallel == True:
                parallel_model = multi_gpu_model(model, gpus=2)
                parallel_model.compile(loss=lda_loss(opti_eigenvalue_num, margin),
                                       optimizer='adam')

                if_datagene = True
                if if_datagene == True:

                    history = parallel_model.fit(datage, epochs=epochs, validation_data=validation_data,
                                                 verbose=1, shuffle=True,callbacks=[earlystop])
                else:
                    history = parallel_model.fit(xt, yt, validation_data=validation_data,
                                                 epochs=epochs, batch_size=100000,  # batch_size=len(xt),#
                                                 shuffle=True, verbose=0, callbacks=[earlystop])
            else:

                model.compile(loss=lda_loss(opti_eigenvalue_num, margin),
                              optimizer='adam')

                if_datagene = True
                if if_datagene == True:

                    history = model.fit(datage, epochs=epochs, validation_data=validation_data,
                                        verbose=1, shuffle=True,callbacks=[earlystop])
                else:

                    history = model.fit(xt, yt, validation_data=validation_data,
                                        epochs=epochs, batch_size=xt.shape[0],  # batch_size=len(xt),#
                                        shuffle=True, verbose=0, callbacks=[earlystop])

            # callbacks=earlystop
        elif if_type == 'sklearn':
            # 创建网络模型
            if layer_num > 1:
                hidden_network = tuple([units for _ in range(layer_num-2)] + [outdim])
                model = MLPRegressor(hidden_layer_sizes=hidden_network, batch_size=len(yt),
                                     max_iter=epochs,early_stopping=False,tol=1,
                                     validation_data=(xv,yv),shuffle=True,alpha=reg_par,
                                     loss='ap', verbose=True, if_hidden_output=True)

            else:
                model = MLPRegressor(hidden_layer_sizes=(units, outdim),batch_size=len(yt),
                                     max_iter=epochs,early_stopping=False,tol=1,
                                     validation_data=(xv,yv),shuffle=True,alpha=reg_par,
                                     loss='ap', verbose=True, if_hidden_output=True)
            # early_stopping=True,validation_fraction=0.2,
            # 默认：loss='squared_loss',
            model.fit(xt, yt)

            #loss = model.loss_curve_

        print('-------------聚类评价------------')
        # 预测
        xtp = model.predict(xt)
        xvp = model.predict(xv)

        if_lda = True
        if if_lda == True:
            lda = LinearDiscriminantAnalysis()
            lda.fit(xtp, yt)
            xtp = lda.transform(xtp)
            xvp = lda.transform(xvp)

        # svm引导训练帮助AP聚类
        if_eval_type = 'ap'

        if if_eval_type == 'ap':
            # 训练集
            if_less_sample = True
            test_size = 0.4
            if if_less_sample == True:
                sss = StratifiedShuffleSplit(test_size=test_size)
                for si, sj in sss.split(xvp, yv):
                    xtpi, yti = xvp[sj], yv[sj]
                    break
            else:
                xtpi, yti = xtp,yt

            ap = AffinityPropagation()
            ap.fit(xtpi)
            yt_pred = ap.labels_

            # 评价
            v = metrics.v_measure_score(yti, yt_pred)
            ars = metrics.adjusted_rand_score(yti, yt_pred)
            # print('调整兰德系数',ars)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            ami = metrics.adjusted_mutual_info_score(yti, yt_pred)
            # print('调整互信息', ami)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            score_train = sum([v, ars, ami]) / 3
            print('score_train', score_train, 'cluster_num', len(np.unique(yt_pred)))

            # 验证集
            if if_less_sample == True:
                sss = StratifiedShuffleSplit(test_size=test_size)
                for si, sj in sss.split(xvp, yv):
                    xvpi, yvi = xvp[sj], yv[sj]
                    break
            else:
                xvpi, yvi = xvp, yv

            ap = AffinityPropagation()
            ap.fit(xvpi)
            yv_pred = ap.labels_

            # 评价
            v = metrics.v_measure_score(yvi, yv_pred)
            ars = metrics.adjusted_rand_score(yvi, yv_pred)
            # print('调整兰德系数',ars)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            ami = metrics.adjusted_mutual_info_score(yvi, yv_pred)
            # print('调整互信息', ami)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            score_test = sum([v, ars, ami]) / 3
            print('score_test', score_test, 'cluster_num', len(np.unique(yv_pred)))

            score_check = (score_train + score_test) / 2
            # score_check = score_test

            # 保存模型
            if score_check > 0.99:
                if if_type == 'keras':
                    model.save('../model/hyperopt/hyperopt_model_cv%s_'
                               '%0.2f_u%s-l%s-r%s-od%s-op%s'
                               '-e%s.h5' % (cv, score_check,
                                           units, layer_num,
                                           reg_par, outdim,
                                           opti_eigenvalue_num,
                                            epochs))
                    if if_scaler == True:
                        joblib.dump(scaler, '../model/hyperopt/hyperopt_std_cv%s_%0.2f_u%s-l%s-r%s-od%s-op%s.pkl' % (cv, score_check,
                                                                                                   units, layer_num,
                                                                                                   reg_par,
                                                                                                   outdim,
                                                                                                   opti_eigenvalue_num))

                    if if_lda == True:
                        joblib.dump(lda, '../model/hyperopt/hyperopt_lda_cv%s_%0.2f_u%s-l%s-r%s-od%s-op%s.pkl' % (cv, score_check,
                                                                                                    units,
                                                                                                    layer_num,
                                                                                                    reg_par,
                                                                                                    outdim,
                                                                                                    opti_eigenvalue_num))
        elif if_eval_type == 'svm':
            clf = svm.SVC(C=0.1, verbose=True)
            clf.fit(xtp, yt)

            yt_pred = clf.predict(xtp)
            train_acc = metrics.accuracy_score(yt, yt_pred)

            yv_pred = clf.predict(xvp)
            valid_acc = metrics.accuracy_score(yv, yv_pred)

            score_check = (train_acc + valid_acc) / 2

            '''
            # ap聚类表现
            ap = AffinityPropagation()
            ap.fit(xvp)
            y_pred = ap.labels_

            # 评价
            v = metrics.v_measure_score(yv, y_pred)
            ars = metrics.adjusted_rand_score(yv, y_pred)
            # print('调整兰德系数',ars)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            ami = metrics.adjusted_mutual_info_score(yv, y_pred)
            # print('调整互信息', ami)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            print('ap聚类表现',sum([v, ars, ami]) / 3)
            '''
            # 保存模型
            if score_check > 0.98:
                if if_type == 'keras':
                    model.save('../model/hyperopt_model_cv%s_'
                               '%0.2f_u%s-l%s-r%s-od%s-op%s'
                               '-e%s.h5' % (cv, score_check,
                                            units, layer_num,
                                            reg_par, outdim,
                                            opti_eigenvalue_num,
                                            epochs))
                    if if_scaler == True:
                        joblib.dump(scaler,
                                    '../model/hyperopt_std_cv%s_%0.2f_u%s-l%s-r%s-od%s-op%s.pkl' % (cv, score_check,
                                                                                                    units, layer_num,
                                                                                                    reg_par,
                                                                                                    outdim,
                                                                                                    opti_eigenvalue_num))

                    if if_lda == True:
                        joblib.dump(lda,
                                    '../model/hyperopt_lda_cv%s_%0.2f_u%s-l%s-r%s-od%s-op%s.pkl' % (cv, score_check,
                                                                                                    units,
                                                                                                    layer_num,
                                                                                                    reg_par,
                                                                                                    outdim,
                                                                                                    opti_eigenvalue_num))

        elif if_eval_type == 'ap_small_size':
            if_algo = 'kmeans'
            if if_algo == 'ap':
                ap = AffinityPropagation()
            elif if_algo == 'kmeans':
                ap = KMeans(n_clusters=num_class)
            ap.fit(xtp)
            yt_pred = ap.labels_

            # 评价
            v = metrics.v_measure_score(yt, yt_pred)
            ars = metrics.adjusted_rand_score(yt, yt_pred)
            # print('调整兰德系数',ars)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            ami = metrics.adjusted_mutual_info_score(yt, yt_pred)
            # print('调整互信息', ami)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            score_train = sum([v, ars, ami]) / 3
            print('score_train', score_train, 'cluster_num', len(np.unique(yt_pred)))

            # 验证集
            if if_algo == 'ap':
                ap = AffinityPropagation()
            elif if_algo == 'kmeans':
                ap = KMeans(n_clusters=num_class)
            ap.fit(xvp)
            yv_pred = ap.labels_

            # 评价
            v = metrics.v_measure_score(yv, yv_pred)
            ars = metrics.adjusted_rand_score(yv, yv_pred)
            # print('调整兰德系数',ars)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            ami = metrics.adjusted_mutual_info_score(yv, yv_pred)
            # print('调整互信息', ami)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            score_test = sum([v, ars, ami]) / 3
            print('score_test', score_test, 'cluster_num', len(np.unique(yv_pred)))

            score_check = (score_train + score_test) / 2

            # score_check = score_test

            # 保存模型
            if score_check > 0.98:
                if if_type == 'keras':
                    model.save('../model/hyperopt_model_cv%s_'
                               '%0.2f_u%s-l%s-r%s-od%s-op%s'
                               '-e%s.h5' % (cv, score_check,
                                           units, layer_num,
                                           reg_par, outdim,
                                           opti_eigenvalue_num,
                                            epochs))
                    if if_scaler == True:
                        joblib.dump(scaler, '../model/hyperopt_std_cv%s_%0.2f_u%s-l%s-r%s-od%s-op%s.pkl' % (cv, score_check,
                                                                                                   units, layer_num,
                                                                                                   reg_par,
                                                                                                   outdim,
                                                                                                   opti_eigenvalue_num))

                    if if_lda == True:
                        joblib.dump(lda, '../model/hyperopt_lda_cv%s_%0.2f_u%s-l%s-r%s-od%s-op%s.pkl' % (cv, score_check,
                                                                                                    units,
                                                                                                    layer_num,
                                                                                                    reg_par,
                                                                                                    outdim,
                                                                                                    opti_eigenvalue_num))

        scores += score_check

        epochs_actual += len(history.history['loss'])

        cv += 1

    arg['epochs'] = epochs_actual / cv

    return - scores / cv  # 最优为-1

# train the DLDA+AP model using the optimized haperparameters
# 根据寻优结果训练模型
def train_model(xt,yt,xv,yv,arg):
    units = arg['units']
    layer_num = arg['layer_num']
    reg_par = arg['reg_par']
    outdim = arg['outdim']
    opti_eigenvalue_num = arg['opti_eigenvalue_num']
    margin = arg['margin']
    epochs = arg['epochs']

    if opti_eigenvalue_num > outdim / 3 * 2:
        opti_eigenvalue_num = int(outdim / 3 * 2)

    # 构建网络
    if_type = 'keras'
    if if_type == 'keras':
        x_dim = xt.shape[1]
        datage = Datagenerator(xt, yt, 50 * 156)
        model = Sequential()
        if_batchnormal = True
        if if_batchnormal == True:
            model.add(Dense(units, input_shape=(x_dim,), activation='relu',
                            kernel_regularizer=l2(reg_par)))

            model.add(BatchNormalization())
            if layer_num >= 2:
                for i in range(layer_num - 2):
                    model.add(Dense(units, activation='relu', kernel_regularizer=l2(reg_par)))
                    model.add(BatchNormalization())

            model.add(Dense(outdim, activation='linear', kernel_regularizer=l2(reg_par)))
            model.add(BatchNormalization())
        else:
            model.add(Dense(units, input_shape=(x_dim,), activation='relu',
                            kernel_regularizer=l2(reg_par)))

            if layer_num >= 2:
                for i in range(layer_num - 2):
                    model.add(Dense(units, activation='relu', kernel_regularizer=l2(reg_par)))

            model.add(Dense(outdim, activation='linear', kernel_regularizer=l2(reg_par)))

        # model.summary()
        earlystop = callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=1, patience=100)

        if_parallel = False
        if if_parallel == True:
            parallel_model = multi_gpu_model(model, gpus=2)
            parallel_model.compile(loss=lda_loss(opti_eigenvalue_num, margin),
                                   optimizer='adam')

            if_datagene = True
            if if_datagene == True:

                history = parallel_model.fit(datage, epochs=epochs,  validation_data=(xv,yv),
                                             verbose=1, shuffle=True)
            else:
                history = parallel_model.fit(xt, yt,  validation_data=(xv, yv),
                                             epochs=epochs, batch_size=100000,  # batch_size=len(xt),#
                                             shuffle=True, verbose=0)  # , callbacks=[earlystop])
        else:

            model.compile(loss=lda_loss(opti_eigenvalue_num, margin),
                          optimizer='adam')

            if_datagene = True
            if if_datagene == True:

                history = model.fit(datage, epochs=epochs, validation_data=(xv, yv),
                                    verbose=1, shuffle=True)
            else:

                history = model.fit(xt, yt, validation_data=(xv, yv),
                                    epochs=epochs, batch_size=50000,  # batch_size=len(xt),#
                                    shuffle=True, verbose=0)  # , callbacks=[earlystop])

        # callbacks=earlystop
        loss = history.history['loss']
        plt.figure()
        epo = [e for e in range(len(loss))]
        plt.plot(epo,loss)
        plt.title('loss with epochs')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()

    # 预测
    xtp = model.predict(xt)
    xvp = model.predict(xv)

    if_lda = True
    if if_lda == True:
        lda = LinearDiscriminantAnalysis()
        lda.fit(xtp, yt)
        xtp = lda.transform(xtp)
        xvp = lda.transform(xvp)

    if_eval_type = 'ap'
    if if_eval_type == 'ap':
        if_sample = True
        if if_sample == True:
            # 训练集
            sss = StratifiedShuffleSplit(test_size=0.5)
            for si, sj in sss.split(xvp, yv):
                xtpi, yti = xvp[sj], yv[sj]
                break

            ap = AffinityPropagation()
            ap.fit(xtpi)
            yt_pred = ap.labels_

            # 评价
            v = metrics.v_measure_score(yti, yt_pred)
            ars = metrics.adjusted_rand_score(yti, yt_pred)
            # print('调整兰德系数',ars)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            ami = metrics.adjusted_mutual_info_score(yti, yt_pred)
            # print('调整互信息', ami)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            score_train = sum([v, ars, ami]) / 3
            print('score_train', score_train, 'cluster_num', len(np.unique(yt_pred)))

            # 验证集

            sss = StratifiedShuffleSplit(test_size=0.5)
            for si, sj in sss.split(xvp, yv):
                xvpi, yvi = xvp[sj], yv[sj]
                break

            ap = AffinityPropagation()
            ap.fit(xvpi)
            yv_pred = ap.labels_

            # 评价
            v = metrics.v_measure_score(yvi, yv_pred)
            ars = metrics.adjusted_rand_score(yvi, yv_pred)
            # print('调整兰德系数',ars)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            ami = metrics.adjusted_mutual_info_score(yvi, yv_pred)
            # print('调整互信息', ami)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            score_test = sum([v, ars, ami]) / 3
            print('score_test', score_test, 'cluster_num', len(np.unique(yv_pred)))

        else:
            ap = AffinityPropagation()
            ap.fit(xtp)
            yt_pred = ap.labels_

            # 评价
            v = metrics.v_measure_score(yt, yt_pred)
            ars = metrics.adjusted_rand_score(yt, yt_pred)
            # print('调整兰德系数',ars)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            ami = metrics.adjusted_mutual_info_score(yt, yt_pred)
            # print('调整互信息', ami)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            score_train = sum([v, ars, ami]) / 3
            print('score_train',score_train,'cluster_num',len(np.unique(yt_pred)))

            ap = AffinityPropagation()
            ap.fit(xvp)
            yv_pred = ap.labels_

            # 评价
            v = metrics.v_measure_score(yv, yv_pred)
            ars = metrics.adjusted_rand_score(yv, yv_pred)
            # print('调整兰德系数',ars)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            ami = metrics.adjusted_mutual_info_score(yv, yv_pred)
            # print('调整互信息', ami)  # [−1,1]，值越大意味着聚类结果与真实情况越吻合
            score_test = sum([v, ars, ami]) / 3
            print('score_test',score_test,'cluster_num',len(np.unique(yv_pred)))

        return model,lda,score_train,score_test

# test the trained model
# 测试训练的模型
def test_model(x_test,y_test,lda,model):
    xt = lda.transform(model.predict(x_test))
    y_pred = ap_cluster(xt)[0]
    num_cluster = len(np.unique(y_pred))

    score_check = eval_index(x_test,xt,y_test,y_pred)[-1]

    return num_cluster, score_check


if __name__ == '__main__':
    x = np.load('../data/label_all_x_160c_sample367.npy')
    y = np.load('../data/label_all_str-y_160c_sample367.npy')

    y = np.array(label_string2number(y))
    num_class = len(np.unique(y))

    # division: train/test set
    # 划分训练集和测试集:1/4为测试集 data 4807 train 3591 test 1216
    # shuffle=False 数据不打乱：保证模型的可重复性
    skf = StratifiedKFold(n_splits=4, shuffle=False)
    train_index = []
    test_index = []
    for i,j  in skf.split(x, y):
        train_index.append(i)
        test_index.append(j)

    num_select = 0
    x_train, x_test = x[train_index[num_select]], x[test_index[num_select]]
    y_train, y_test = y[train_index[num_select]], y[test_index[num_select]]

    # data Standardization: no
    # 数据标准化: 不进行std
    if_scaler = False
    if if_scaler == True:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        joblib.dump(scaler,'../model/hyperopt_std.pkl')

    # optimization of hyperparameters: Bayesian optimization
    # 超参数寻优：贝叶斯优化
    # 目标函数 opt_ob
    # 构建寻优空间
    if_hyperopt = False
    if if_hyperopt == True:
        reg = [0,1e-1,1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        space = {
            'units': hp.randint('units', 48, 2048),
            'layer_num': hp.randint('layer_num', 2, 10),
            'reg_par': hp.choice('reg_par', reg),
            'outdim': hp.randint('outdim', 24, 1024),
            'epochs': hp.randint('epochs', 5, 3000),  # 40
            'margin': hp.uniform('margin', 1, 1e5), # 实际设为0, not used in loss function actually
            'opti_eigenvalue_num': hp.randint('opti_eigenvalue_num', 1,num_class-1)
        }

        trials = Trials()
        best = fmin(fn=opt_ob, space=space, algo=tpe.suggest,
                    max_evals=200, trials=trials)

        best['opti_eigenvalue_num'] = min(best['outdim'], num_class - 1,
                                          best['opti_eigenvalue_num'])
        best['reg_par'] = reg[best['reg_par']]

        name = list(space.keys())
        values = [t['result']['loss'] for t in trials.trials]

        print('result',min(values),'arg',best)
        if_save = False
        if if_save == True:
            np.save('../model/hyperopt/hyperopt_bestpara_score%0.2f.npy'%min(values),best)
            np.save('../model/hyperopt/hyperopt_trials-trials_score%0.2f.npy'%min(values),trials.trials)

        # visualize the process of optimization
        # 寻优过程可视化
        # 变量随时间t取值
        if_view_t = False
        if if_view_t == True:
            for i in range(len(name)):
                plt.figure()
                xs = [t['tid'] for t in trials.trials]
                ys = [t['misc']['vals'][name[i]] for t in trials.trials]
                plt.xlim(xs[0], xs[-1])
                plt.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
                plt.title('$%s$ $vs$ $t$ ' % name[i], fontsize=18)
                plt.xlabel('$t$', fontsize=16)
                plt.ylabel('$%s$' % name[i], fontsize=16)
                plt.show()

        # 变量取值与优化目标函数值
        if_view_val = False
        if if_view_val == True:
            for i in range(len(name)):
                plt.figure()
                xs = [t['misc']['vals'][name[i]] for t in trials.trials]
                ys = [t['result']['loss'] for t in trials.trials]
                # plt.xlim((xs[0], xs[-1]))
                plt.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
                plt.title('$val$ $vs$ $%s$ ' % name[i], fontsize=18)
                plt.ylabel('$val$', fontsize=16)
                plt.xlabel('$%s$' % name[i], fontsize=16)
                plt.show()

    # if load the optimized hyperparameters saved before
    if_load_best = False
    if if_load_best == True:
        best = np.load('../model/hyperopt_ap-test-size-1_20200814/hyperopt_bestpara_score-0.80.npy',
                       allow_pickle=True)
        best = dict(best.tolist())
        print(best)
        trials = np.load('../model/hyperopt_ap-test-size-1_20200814/hyperopt_trials-trials_score-0.80.npy',
                         allow_pickle=True)

    # train the DLDA+AP model using specific (optimized) hyperparameters on the whole training set
    # and test it on test set
    # 利用寻优参数训练所有训练集数据并测试
    filepath = '../model/hyperopt'
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    if_train_test = True
    if if_train_test == True:
        print('测试目前最优参数')
        best = {'outdim': 1024,
                'epochs': 1950,  # 160
                'layer_num': 3,
                'units': 2048,
                'opti_eigenvalue_num': 120,
                'margin': 1.0,
                'reg_par': 0}

        r = train_model(x_train,y_train,x_test,y_test,best)
        model, lda, score_train, score_test = r
        model.save('../model/hyperopt/hyperopt_model_0.99_u1144-l3-r1e-05-od456-op5-e2941.h5')
        joblib.dump(lda,'../model/hyperopt/hyperopt_lda_0.99_u1144-l3-r1e-05-od456-op5.pkl')
        np.save('../model/hyperopt/hyperopt_lda_0.99_u1144-l3-r1e-05-od456-op5_score-train-test.npy',
                [score_train,score_test])

    # (batch) test the trained model saved during the hyperparameter optimization process
    # 对寻优过程中保存的模型进行批量测试
    if_check_model = False
    if if_check_model == True:
        # 获取文件名：lda和model
        filename_lda = []
        filename_model = []
        filepath = '../model/hyperopt'
        for dirname, _, filenames in os.walk(filepath):
            filenames.sort()
            for filename in filenames:
                if 'lda' in filename:
                    filename_lda.append(os.path.join(filepath, filename))
                elif 'model' in filename:
                    filename_model.append(os.path.join(filepath, filename))

        # 批量测试
        score = []
        cluster_num = []
        for i in range(0,len(filename_lda),3):
            lda = joblib.load(filename_lda[i])
            model = load_model(filename_model[i],compile=False)
            num_cluster,score_check = test_model(x_test,y_test,lda,model)
            score.append(score_check)
            cluster_num.append(num_cluster)
            print('测试模型',i,'\n',
                  filename_lda[i],'\n',
                  filename_model[i],'\n',
                  '聚类数：',num_cluster,
                  '模型评价',score_check,'\n')
        np.save('../model/hyperopt/test_model_score.npy',score)
        np.save('../model/hyperopt/test_model_cluster_num.npy', cluster_num)
        np.save('../model/hyperopt/test_model_lda_filename.npy', filename_lda)
        np.save('../model/hyperopt/test_model_model_filename.npy', filename_model)