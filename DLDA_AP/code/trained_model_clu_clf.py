#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import joblib
from tensorflow.keras.models import load_model

# visualization format
def view_format():

    _hour = list(range(1,25))

    x_lines = [1,2.5,5.5,7.5,10.5,13.5,16.5,19.5,22.5,24]
    for xi in x_lines:
        plt.axvline(xi,color='gray', linestyle='--')

    aver = 100 / 24
    m_low = aver * 0.5  # 2
    m_high = aver * 1.5  # 6

    plt.axhline(aver, color='darkred', linestyle='-.')
    plt.axhline(m_low, color='r', linestyle='--')
    plt.axhline(m_high, color='r', linestyle='--')

    y_text = 0.5
    x_text = [1.3,3.6,6.1,8.6,11.6,14.6,17.6,20.5,22.8]
    for i in range(len(x_text)):
        plt.text(x_text[i], y_text, 't-%s'%(i+1), color='brown')

    plt.xlabel('Time (hour)',{'size': 12})
    plt.ylabel('Percentage (%)',{'size': 12})

    plt.xticks(_hour)


if __name__ == '__main__':
    # sklearn.__version__ : 0.21
    # tensorflow.__version__: 2.0

    # load trained model
    dnn = load_model('../model/trained_dnn.h5',compile=False)
    lda = joblib.load('../model/trained_lda.pkl')

    # load DECP samples
    y = np.load('../data/example_y_c160-10.npy', allow_pickle=True)
    x = np.load('../data/example_x_c160-10.npy', allow_pickle=True)

    # DLDA+AP
    x_dlda = lda.transform(dnn.predict(x))

    # clustering
    # 聚类
    if_clu = True
    if if_clu == True:
        ap = AffinityPropagation(max_iter=300).fit(x_dlda)
        y_pred = ap.labels_
        y_true = y
        c = len(np.unique(y_pred))

        # evaluation
        # labeled
        # [−1,1], bigger value means better results
        v = metrics.v_measure_score(y_true, y_pred)
        ari = metrics.adjusted_rand_score(y_true, y_pred)
        ami = metrics.adjusted_mutual_info_score(y_true, y_pred)

        # unlabeled
        # [−1,1], bigger value means better results
        sc = metrics.silhouette_score(x_dlda, y_pred, metric='euclidean')
        # [0, infinite], smaller value means better results
        dbi = metrics.davies_bouldin_score(x_dlda, y_pred)

        print('Number of clusters:%s' %c)
        print('V_score:%0.3f, ARI:%0.3f, AMI:%0.3f' % (v, ari, ami))
        print('SC:%0.3f, DBI:%0.3f' % (sc, dbi))

    # classify the samples to one of the load dictionary patterns
    # 分类
    if_clf = True
    if if_clf == True:
        x_mainstream = np.load('../data/load_dictionary_pattern_mainstream_282.npy',allow_pickle=True)
        x_outlier = np.load('../data/load_dictionary_pattern_outlier_176.npy',allow_pickle=True)
        x_ldp = np.array(x_mainstream.tolist() + x_outlier.tolist())

        y_ldp = list(range(len(x_ldp)))

        x_ldp_dlda = lda.transform(dnn.predict(x_ldp))
        clf = KNeighborsClassifier(1)
        clf.fit(x_ldp_dlda, y_ldp)
        y_pred_clf = clf.predict(x_dlda)

        # visualization
        label = 0
        x_label = [x[i] for i in range(len(x)) if y_pred_clf[i] == label]

        plt.figure(figsize=(6,4.5))
        view_format()
        hours = list(range(1,25))
        for xi in x_label:
            plt.plot(hours,xi,alpha=0.5)
        plt.plot(hours,x_ldp[label],c='black',lw=2)
        plt.ylim([0,20])
        plt.title('Samples classified into load dictionary pattern-%s'%(label+1))
        plt.show()

