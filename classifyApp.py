# coding: utf-8
'''
经过我的计算，每一类文本的训练数据都是一样的共31050条
文本分类有9种，训练集共279450条数据
'''
# from pylab import *
# from matplotlib.font_manager import FontProperties
# 这是为了画图导入的第三方库
from openpyxl import Workbook

import tensorflow as tf
import numpy as np
import os
import time
import datetime

from sklearn import metrics

from classifyUtils import data_process
from text_cnn import TextCNN
import math
from tensorflow.contrib import learn
import jieba


# tf.reset_default_graph()

class classifyApplication:
    def __init__(self, sess, device='/cpu:0'):
        with sess.as_default():
            with sess.graph.as_default():
                self.word_embedings_path = "./data_ai/cbowData/classifyDocument.txt.ebd.npy"
                # self.word_embedings_path = "./data_ai/cbowData/classify.npy"
                self.vocb_path = "./data_ai/cbowData/classify1.vab"
                self.model_path = "./data_ai/classifyModel/1111"
                # self.num_classes = 9
                self.num_classes = 9
                # self.max_sentence_len = 20
                self.max_sentence_len = 50
                # self.embedding_dim = 200
                self.embedding_dim = 128
                # self.filter_sizes = "2,3,4"
                self.filter_sizes = "3,4,5"
                # self.filter_sizes = "4,5,6"
                # self.dropout_keep_prob = 1.0
                self.dropout_keep_prob = 0.8
                self.l2_reg_lambda = 0.8
                # self.num_filters = 128
                self.num_filters = 128
                self.num_checkpoints = 5
                # self.batch_size = 128
                self.batch_size = 128
                #全连接层隐藏个数   这是新加入的代码
                # self.fc_hidden_size = 1024
                # self.num_filters = 100


                self.data_helpers = data_process(
                    train_data_path="./data_ai/classifyData/train_data1.txt",
                #    train_data_path="./data_ai/classifyData/train_add.txt",
                    word_embedings_path=self.word_embedings_path,
                    vocb_path=self.vocb_path,
                    num_classes=self.num_classes,

                    max_document_length=self.max_sentence_len)

                self.data_helpers.load_wordebedding()
                self.data_helpers.load_data()
                self.cnn = TextCNN(
                    w2v_model=self.data_helpers.word_embeddings,
                    sequence_length=self.max_sentence_len,
                    num_classes=self.num_classes,
                    embedding_size=self.embedding_dim,
                    filter_sizes=list(map(int, self.filter_sizes.split(","))),
                    num_filters=self.num_filters,
                    # 这是我新加入的代码
                    # fc_hidden_size=self.fc_hidden_size,
                    l2_reg_lambda=self.l2_reg_lambda,
                    device=device
                )

                self.saver = tf.train.Saver(max_to_keep=self.num_checkpoints)

                # init = tf.initialize_all_variables()
                init = tf.global_variables_initializer()
                sess.run(init)
                # 创建了一个saver对象
                self.loadModel(sess)
    def loadModel(self, sess):

        ckpt = tf.train.get_checkpoint_state(self.model_path)
        print("ckpt", ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            # 恢复变量
            print("restore from history model.")
        else:
            print("there is no classify model.")

    def test(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                #self.loadModel(sess)
                #print("Testing......start")

                data_len = len(self.data_helpers.x_dev)
                num_batch = int((data_len - 1) / self.batch_size) + 1
                y_test_cls = np.argmax(self.data_helpers.y_dev, 1)
                y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)  # 保存预测结果
                for i in range(num_batch):
                    start_id = i * self.batch_size
                    end_id = min((i + 1) * self.batch_size, data_len)
                    feed_dick = {self.cnn.input_x: self.data_helpers.x_dev[start_id:end_id],
                                 self.cnn.input_y: self.data_helpers.y_dev[start_id:end_id],
                                 self.cnn.dropout_keep_prob: self.dropout_keep_prob
                                 }
                    y_pred_cls[start_id:end_id] = sess.run(
                        self.cnn.predictions,
                        feed_dict=feed_dick)
                cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
                test_precision, test_recall, test_f1 = self.data_helpers.evalution(cm)
                print(
                     "test_precision:", test_precision,
                    "test_recall:", test_recall, "test_f1", test_f1)
                print("Testing......end")

    def train(self, sess):
        mybook = Workbook()
        wa = mybook.active
        wa.append(['损失率','精确率'])
        with sess.as_default():
            with sess.graph.as_default():

                tf.summary.scalar('loss', self.cnn.loss)

                tf.summary.scalar("accuracy", self.cnn.accuracy)
                tf.summary.scalar("confusion_matrix", self.cnn.confusion_matrix)
                writer = tf.summary.FileWriter(self.model_path)
                # init = tf.global_variables_initializer()
                # sess.run(tf.local_variables_initializer())
                # sess.run(tf.global_variables_initializer())
                # sess.run(tf.local_variables_initializer())

                writer.add_graph(sess.graph)
                # 原始代码的epoch是12
                for epoch in range(3):
                    print('Epoch:', epoch)
                    total_batch = 0  # 总批数
                    count = 0 # 这是加入的总batch_size个数
                    data_len = len(self.data_helpers.x_train)
                    num_batch = int((data_len - 1) / self.batch_size) + 1
                    data_train = self.data_helpers.batch_iter(self.data_helpers.x_train,self.data_helpers.y_train)
                    for x_batch,y_batch in data_train:
                        # 存放损失率和精确率
                        feed_dick = {self.cnn.input_x: x_batch,
                                     self.cnn.input_y: y_batch,
                                     self.cnn.dropout_keep_prob: self.dropout_keep_prob
                                     }
                        _,loss, confusion_matrix, accuracy = sess.run(
                            [self.cnn.train_step,self.cnn.loss, self.cnn.confusion_matrix, self.cnn.accuracy],
                            feed_dict=feed_dick)
                        # confusion_matrix混淆矩阵，是分类模型中总结预测结果的情形分析表
                        train_precision, train_recall, train_f1 = self.data_helpers.evalution(confusion_matrix)


                        total_batch += 1

                        if total_batch%10==0:
                            count = count + 1
                            # 每10轮的结果放在列表中，然后再存储在excel表格中
                            a = []
                            a.append(loss)
                            a.append(accuracy)
                            wa.append(a)
                            # print(a)
                            mybook.save('C:\\Users\\luminous\\Desktop\\data5.xlsx')

                            print("第", total_batch, "批", "  loss:", loss, "accuracy:", accuracy, "train_precision:",
                              train_precision,"train_recall:", train_recall, "train_f1", train_f1)
                        
                        if (total_batch % 500 == 0):
                            save_path = os.path.join(self.model_path, "best_validation" + str(total_batch))
                            self.saver.save(sess=sess, save_path=save_path)

                        if total_batch%100 ==0:
                            print("第", total_batch, "批", "  loss:", loss, "accuracy:", accuracy)
                            self.test(sess)
        print(count)
        mybook.save('C:\\Users\\luminous\\Desktop\\data.xlsx')


    #
    #
    # def classifyApp(self, sess):
    #     with sess.as_default():
    #         with sess.graph.as_default():
    #             text = "application"
    #             while (text != "" and text != " "):
    #
    #                 text = input("请输入一句话：")
    #                 if text == "quit" or text == "" or text == " ": break
    #                 text = text.strip()
    #                 seg_list = list(jieba.cut(text))
    #                 x_data = self.data_helpers.handle_input(' '.join(seg_list))
    #                 feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
    #                 _predic = sess.run([self.cnn.predictions], feed_dict)
    #                 print("%s is %d" % (text, _predic[0]))

    def classifyApp0(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                count = 0
                #lines = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\0.txt","r",encoding="utf-8").readlines()
                lines = open(".\\data_ai\\classifyData\\0.txt", "r",
                             encoding="utf-8").readlines()
                for line in lines:
                    text = line
                    if text == "quit" or text == "" or text == " ": break
                    text = text.strip()
                    seg_list = list(jieba.cut(text))
                    x_data = self.data_helpers.handle_input(' '.join(seg_list))
                    feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                    _predic = sess.run([self.cnn.predictions], feed_dict)
                    # print(type(_predic[0][0]))
                    # print("%s is %d" % (text, _predic[0]))
                    if _predic[0][0] == 0:
                        count += 1
                print(count)
    def classifyApp1(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                count = 0
                lines = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\1.txt","r",encoding="utf-8").readlines()
                for line in lines:
                    text = line
                    if text == "quit" or text == "" or text == " ": break
                    text = text.strip()
                    seg_list = list(jieba.cut(text))
                    x_data = self.data_helpers.handle_input(' '.join(seg_list))
                    feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                    _predic = sess.run([self.cnn.predictions], feed_dict)
                    # print(type(_predic[0][0]))
                    # print("%s is %d" % (text, _predic[0]))
                    if _predic[0][0] == 1:
                        count += 1
                print(count)
    def classifyApp2(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                count = 0
                lines = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\2.txt","r",encoding="utf-8").readlines()
                for line in lines:
                    text = line
                    if text == "quit" or text == "" or text == " ": break
                    text = text.strip()
                    seg_list = list(jieba.cut(text))
                    x_data = self.data_helpers.handle_input(' '.join(seg_list))
                    feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                    _predic = sess.run([self.cnn.predictions], feed_dict)
                    # print(type(_predic[0][0]))
                    # print("%s is %d" % (text, _predic[0]))
                    if _predic[0][0] == 2:
                        count += 1
                print(count)
    def classifyApp3(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                count = 0
                lines = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\3.txt","r",encoding="utf-8").readlines()
                for line in lines:
                    text = line
                    if text == "quit" or text == "" or text == " ": break
                    text = text.strip()
                    seg_list = list(jieba.cut(text))
                    x_data = self.data_helpers.handle_input(' '.join(seg_list))
                    feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                    _predic = sess.run([self.cnn.predictions], feed_dict)
                    # print(type(_predic[0][0]))
                    # print("%s is %d" % (text, _predic[0]))
                    if _predic[0][0] == 3:
                        count += 1
                print(count)
    def classifyApp4(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                count = 0
                lines = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\4.txt","r",encoding="utf-8").readlines()
                for line in lines:
                    text = line
                    if text == "quit" or text == "" or text == " ": break
                    text = text.strip()
                    seg_list = list(jieba.cut(text))
                    x_data = self.data_helpers.handle_input(' '.join(seg_list))
                    feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                    _predic = sess.run([self.cnn.predictions], feed_dict)
                    # print(type(_predic[0][0]))
                    # print("%s is %d" % (text, _predic[0]))
                    if _predic[0][0] == 4:
                        count += 1
                print(count)
    def classifyApp5(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                count = 0
                lines = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\5.txt","r",encoding="utf-8").readlines()
                for line in lines:
                    text = line
                    if text == "quit" or text == "" or text == " ": break
                    text = text.strip()
                    seg_list = list(jieba.cut(text))
                    x_data = self.data_helpers.handle_input(' '.join(seg_list))
                    feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                    _predic = sess.run([self.cnn.predictions], feed_dict)
                    # print(type(_predic[0][0]))
                    # print("%s is %d" % (text, _predic[0]))
                    if _predic[0][0] == 5:
                        count += 1
                print(count)
    def classifyApp6(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                count = 0
                lines = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\6.txt","r",encoding="utf-8").readlines()
                for line in lines:
                    text = line
                    if text == "quit" or text == "" or text == " ": break
                    text = text.strip()
                    seg_list = list(jieba.cut(text))
                    x_data = self.data_helpers.handle_input(' '.join(seg_list))
                    feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                    _predic = sess.run([self.cnn.predictions], feed_dict)
                    # print(type(_predic[0][0]))
                    # print("%s is %d" % (text, _predic[0]))
                    if _predic[0][0] == 6:
                        count += 1
                print(count)
    def classifyApp7(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                count = 0
                lines = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\7.txt","r",encoding="utf-8").readlines()
                for line in lines:
                    text = line
                    if text == "quit" or text == "" or text == " ": break
                    text = text.strip()
                    seg_list = list(jieba.cut(text))
                    x_data = self.data_helpers.handle_input(' '.join(seg_list))
                    feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                    _predic = sess.run([self.cnn.predictions], feed_dict)
                    # print(type(_predic[0][0]))
                    # print("%s is %d" % (text, _predic[0]))
                    if _predic[0][0] == 7:
                        count += 1
                print(count)
    def classifyApp8(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                count = 0
                lines = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\8.txt","r",encoding="utf-8").readlines()
                for line in lines:
                    text = line
                    if text == "quit" or text == "" or text == " ": break
                    text = text.strip()
                    seg_list = list(jieba.cut(text))
                    x_data = self.data_helpers.handle_input(' '.join(seg_list))
                    feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                    _predic = sess.run([self.cnn.predictions], feed_dict)
                    # print(type(_predic[0][0]))
                    # print("%s is %d" % (text, _predic[0]))
                    if _predic[0][0] == 8:
                        count += 1
                print(count)

    def questionClassify(self, sess, text):
        with sess.as_default():
            with sess.graph.as_default():
                text = text.strip()
                # print("text:", text)
                seg_list = list(jieba.cut(text))
                # print("seg_list:", seg_list)
                x_data = self.data_helpers.handle_input(' '.join(seg_list))
                # print("x_data", x_data)
                feed_dict = {self.cnn.input_x: x_data, self.cnn.dropout_keep_prob: self.dropout_keep_prob}
                # print("feed_dict:", feed_dict)
                _predic = sess.run([self.cnn.predictions], feed_dict)
                # print("_predic", _predic)
                return _predic[0]


if __name__ == "__main__":

    graph = tf.Graph()
    # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加
    log_device_placement = True  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session_conf = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=allow_soft_placement,
                                  log_device_placement=log_device_placement)

    sess = tf.Session(graph=graph, config=session_conf)


    classifyApp = classifyApplication(sess)
    classifyApp.train(sess)
    # classifyApp.loadModel(sess)
    # classifyApp.test(sess)
    '''
    classifyApp.classifyApp0(sess)
    classifyApp.classifyApp1(sess)
    classifyApp.classifyApp2(sess)
    classifyApp.classifyApp3(sess)
    classifyApp.classifyApp4(sess)
    classifyApp.classifyApp5(sess)
    classifyApp.classifyApp6(sess)
    classifyApp.classifyApp7(sess)
    classifyApp.classifyApp8(sess)
    '''



