#coding=utf-8


import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,w2v_model, sequence_length, num_classes,
                 embedding_size, filter_sizes,
                 num_filters,l2_reg_lambda=0.0,device = '/cpu:0'):
        # 词向量
        self.word_embeddings = w2v_model
        # 句子长度
        self.sequence_length = sequence_length
        # 分类标签总数
        self.num_classes = num_classes
        # 模型维度
        self.embedding_size = embedding_size
        # 卷积核尺度
        self.filter_sizes = filter_sizes
        # 每一个类卷积核数量
        self.num_filters = num_filters

        # 正则化
        self.l2_reg_lambda = l2_reg_lambda




        # Placeholders for input, output and dropout
        # self.input_x = tf.placeholder(tf.int32,[None,None],name="input_x")
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        # 这句代码什么意思？？？？
        l2_loss = tf.constant(0.0)

        with tf.device(device):
            # Embedding layer
            word_embeddings = tf.Variable(initial_value=self.word_embeddings, trainable=True)  # lookup table
            self.embedded_chars = tf.nn.embedding_lookup(word_embeddings, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),dtype=tf.float32, name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]),dtype=tf.float32, name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # 非线性激活函数

                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = self.num_filters * len(self.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1,num_filters_total])
            # 从这里开始，是我新加入的代码

            # # Fully Connected layer
            # with tf.name_scope("fc"):
            #     W = tf.Variable(tf.truncated_normal(shape=[num_filters_total,fc_hidden_size],stddev=0.1
            #                                         ,dtype=tf.float32),name="W")
            #     b = tf.Variable(tf.constant(0.1,shape=[fc_hidden_size],dtype=tf.float32),name="b")
            #     self.fc = tf.nn.xw_plus_b(self.h_pool_flat,W,b)
            #     # 添加非线性函数
            #     self.fc_out = tf.nn.relu(self.fc,name="relu")
            # # Add dropout
            # with tf.name_scope("dropout"):
            #     self.h_drop = tf.nn.dropout(self.fc_out,self.dropout_keep_prob)
            # # Final scores
            # with tf.name_scope("output"):
            #     W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size,num_classes],
            #                                         stddev=0.1,dtype=tf.float32),name="W")
            #     b = tf.Variable(tf.constant(0.1,shape=[num_classes],dtype=tf.float32),name="b")
            #     self.logits = tf.nn.xw_plus_b(self.h_drop,W,b,name="logits")
            #     self.scores = tf.sigmoid(self.logits,name="scores")
            #     self.predictions = tf.round(self.scores,name="predictions")
            #
            # # Calculate mean cross-entroy loss L2 loss 计算交叉熵损失
            # with tf.name_scope("loss"):
            #     losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            #     losses = tf.reduce_mean(tf.reduce_sum(losses,axis=0),name="sigmod_losses")
            #     # l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v,tf.float32)) for v in tf.trainable_variables()],
            #     #                      name="l2_losses") * self.l2_reg_lambda
            #     l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v,tf.float64)) for v in tf.trainable_variables()],
            #                          name="l2_losses") * self.l2_reg_lambda
            #
            #     self.loss = tf.add(losses,l2_losses,name="loss")
            #
            # with tf.name_scope("preformance"):
            #     self.precision = tf.metrics.precision(self.input_y,self.predictions,name="precision-micro")[1]
            #     self.recall = tf.metrics.recall(self.input_y,self.predictions,name="recall-micro")[1]
            #








            # Add dropout，以概率1-dropout_keep_prob，随机丢弃一些节点
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                            "W",
                            shape=[num_filters_total, self.num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss
                self.train_step = tf.train.AdamOptimizer(1e-4).minimize(losses)
  #
            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            #Precission Recall F1_score
            with tf.name_scope("confusion_matrix"):
                self.confusion_matrix = tf.contrib.metrics.confusion_matrix(self.predictions, tf.argmax(self.input_y, 1), num_classes=self.num_classes,dtype=tf.int32, name="confusion_matrix", weights=None)
