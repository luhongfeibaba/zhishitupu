
# coding: utf-8

import  tensorflow as tf
from  tensorflow.contrib import  crf
from Parameters import Parameters as pm
import numpy as np
from nerUtils import DATAPROCESS
import  random
tf.reset_default_graph()


class BiLSTM_CRF(object):

    def __init__(self, batch_size, tag_nums, hidden_nums, sentence_len, word_embeddings, device='/cpu:0'):
        self.batch_size = batch_size
        self.tag_nums = tag_nums
        self.hidden_nums = hidden_nums
        self.sentence_len = sentence_len
        self.word_embeddings = word_embeddings
        # 新加入的学习率
        self.device = device

        with tf.device(device):
            # 网络的变量
            word_embeddings = tf.Variable(initial_value=word_embeddings, trainable=True)  # 参与训练
            # 输入占位符
            # 代表输入是二维矩阵，行不确定，列是句子长度
            self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.sentence_len], name='input_word_id')#输入词的id
            self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.sentence_len], name='input_labels')
            self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_lengths_vector')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            # 这个dropout_keep_pro的意思：训练时删除一部分训练样本，删除的比例就是它，防止过拟合
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            with tf.name_scope('projection'):
                # 投影层,先将输入的词投影成相应的词向量
                word_id = self.input_x
                word_vectors = tf.nn.embedding_lookup(word_embeddings, ids=word_id, name='word_vectors')
                # word_vectors = tf.nn.dropout(word_vectors,0.8)
            with tf.name_scope('bi-lstm'):

                # labels = tf.reshape(input_y,shape=[-1,self.sentence_len],name='labels')
                # labels = tf.reshape(input_y,shape=[-1,self.tag_nums],name='labels')
                labels = tf.reshape(self.input_y,shape=[self.batch_size,self.sentence_len],name='labels')
                fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_nums)
                bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_nums)

                # outputs三维张量，[batchsize,seq_length,2*hidden_dim],
                # 双向传播
                output, _state = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell,bw_lstm_cell,inputs=word_vectors,sequence_length=self.sequence_lengths,dtype=tf.float32)
                # print('output, _state: ', output, _state)
                fw_output = output[0]  # [batch_size,self.sentence_len,self.hidden_nums]
                bw_output = output[1]  # [batch_size,self.sentence_len,self.hidden_nums]
                V1 = tf.get_variable('V1',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[self.hidden_nums,self.hidden_nums])
                V2 = tf.get_variable('V2',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=[self.hidden_nums,self.hidden_nums])
                fw_output = tf.reshape(tf.matmul(tf.reshape(fw_output,[-1,self.hidden_nums],name='Lai') , V1),shape=tf.shape(output[0]))
                bw_output = tf.reshape(tf.matmul( tf.reshape(bw_output,[-1,self.hidden_nums],name='Rai') , V2),shape=tf.shape(output[1]))
                contact = tf.concat([fw_output,bw_output],-1,name='bi_lstm_concat')#[batch_size,self.sentence_len,2*self.hidden_nums]
                contact = tf.nn.dropout(contact,self.dropout_keep_prob)
                s = tf.shape(contact)
                contact_reshape = tf.reshape(contact, shape=[-1, 2*self.hidden_nums], name='contact')
                W_lstm = tf.get_variable('W_lstm', dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.hidden_nums,self.tag_nums],trainable=True)
                b_lstm = tf.get_variable('b_lstm', initializer=tf.zeros(shape=[self.tag_nums]))
                p = tf.nn.relu(tf.matmul(contact_reshape, W_lstm)+b_lstm)
                self.logit = tf.reshape(p, shape=[-1, self.sentence_len, self.tag_nums], name='omit_matrix')

            with tf.name_scope("crf"):
                log_likelihood, transition_matrix = crf.crf_log_likelihood(self.logit, labels, sequence_lengths=self.sequence_lengths)
                self.crf_labels, _ = crf.crf_decode(self.logit, transition_matrix, sequence_length=self.sequence_lengths) #返回的第一个值:decode_tags: A [batch_size, max_seq_len]

            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(-log_likelihood)  # 最大似然取负，使用梯度下降

            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(pm.learning_rate)
                # print("pm.learning_rate:", pm.learning_rate)
                gradients, variable = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
                self.optimizer = optimizer.apply_gradients(zip(gradients, variable), global_step=self.global_step)

    def feed_data(self, x_batch, y_batch, seq_length, keep_pro):

        feed_dict = {self.input_x: np.array(x_batch),
                     self.input_y: np.array(y_batch),
                     self.sequence_lengths: seq_length,
                     self.dropout_keep_prob: keep_pro
                     }
        return feed_dict
