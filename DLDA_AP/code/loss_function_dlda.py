#!/usr/bin/python
# -*- coding:

'''
reference:
https://github.com/CPJKU/deep_lda       # Theano backend
https://github.com/VahidooX/DeepLDA     # Theano backend
'''

import tensorflow as tf
# tensorflow.__version__: 2.0

def lda_loss(n_components, margin):
    """
    The main loss function (inner_lda_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    i.e. pass parameters (n_components, margin) to loss function (inner_lda_objective)

    由于Keras格式的限制，通过这种函数构造方式可将参数传入损失函数。
    在训练时，调用此损失函数只需指定损失函数为本函数即可。

    model.compile(loss=lda_loss(n_components, margin),optimizer=optimizer)
    """

    def inner_lda_objective(y_true, y_pred):
        """
        It is the loss function of LDA.
        """
        r = 1e-4

        # transfer the data type
        # 转换数值类型
        y_pred = tf.cast(y_pred, tf.float32)

        # calculate the within scatter
        # 计算类内离散度矩阵
        def fn(label_target, label_y, preds):
            # 获取label_y中元素值为label_target的所有元素索引号
            label_i_indexes = tf.where(tf.equal(label_y, label_target))
            label_i_indexes = tf.reshape(label_i_indexes, (1, -1))  # 2D==》1D
            # 根据索引label_i_indexes从axis轴收集preds中对应的数据
            X = tf.gather(preds, label_i_indexes, axis=0)[0]
            X_mean = X - tf.reduce_mean(X, axis=0)
            # 样本数
            m = tf.cast(tf.shape(X_mean)[0], tf.float32)
            return (1 / (m - 1)) * tf.matmul(tf.transpose(X_mean), X_mean)

        # get the labels
        # 获取所有类标签
        @tf.function
        def get_label(y):
            # 传入数据：y_true.shape = (None, 1)
            return tf.unique(tf.reshape(y, (1, -1))[0])[0]

        label = get_label(y_true)

        # label:y_true中出现的所有标签（按先后出现顺序排列）
        # y_idx:y_true中的元素对应label中的顺序索引号

        # scan over groups
        # 计算每个类的类内离散度矩阵
        covs_t = tf.map_fn(lambda label_target: fn(label_target, y_true, y_pred),
                           label, dtype=tf.float32)

        # compute average covariance matrix (within scatter)
        Sw_t = tf.reduce_mean(covs_t, axis=0)
        dim = Sw_t.shape[0]  # 特征维度

        # compute total scatter
        Xt_bar = y_pred - tf.reduce_mean(y_pred, axis=0)
        m = tf.cast(tf.shape(Xt_bar)[0], tf.float32)
        St_t = (1 / (m - 1)) * tf.matmul(tf.transpose(Xt_bar), Xt_bar)

        # compute between scatter
        Sb_t = St_t - Sw_t

        # cope for numerical instability (regularize)
        # 加强计算稳定性：在计算逆矩阵时，先给原矩阵加上一个小元素值的单位矩阵
        Sw_t += tf.eye(dim)*r
        inv_Sw = tf.linalg.inv(Sw_t)

        # 协方差矩阵为对称矩阵
        # calculate the eigenvalues of Sw-1*Sb
        evals_t = tf.linalg.eigh(tf.matmul(inv_Sw,Sb_t))[0] # 从小到大排列
        #evals_t = tf.cast(e,tf.float32)

        # calculate the value of loss function: cost
        # 取前 (类别数-1) 个大的特征值计算损失函数值
        # 原因：对神经网络输出再做一次LDA降维后的数据进行聚类
        @tf.function
        def cal_cost(n_components,margin):
            # 调整n_conponents的值
            c_1 = len(label) - 1
            if dim < c_1:
                c_1 = dim

            # 获取前c-1大的特征值
            top_c_1_evals = evals_t[-c_1:]

            if n_components > c_1:
                n_components = c_1

            # 选择c-1个特征值中最小的n_components个特征值
            min_k_evals = top_c_1_evals[:n_components]

            if_use_margin = False
            if if_use_margin == True:
                index_min = tf.argmin(top_c_1_evals, axis=0)
                thresh_min = top_c_1_evals[index_min] + margin

                mask_min = top_c_1_evals < thresh_min
                cost_min = tf.boolean_mask(top_c_1_evals, mask_min)
                cost = - tf.reduce_mean(cost_min)
            else:
                cost = - tf.reduce_mean(min_k_evals)

            return cost

        loss = cal_cost(n_components,margin)

        return tf.cast(loss,tf.float32)

    return inner_lda_objective
