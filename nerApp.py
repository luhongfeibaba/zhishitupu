
# coding: utf-8
import os

import  tensorflow as tf
from  tensorflow.contrib import  crf
from Parameters import Parameters as pm
import  random
from nerUtils import *
import logging
import datetime
from BiLSTM_CRF import BiLSTM_CRF

debug = False
batch_size = 128

class nerAppication:
    #参数
    def __init__(self, sess, device='/cpu:0'):
        tf.reset_default_graph()
        with sess.as_default():
            with sess.graph.as_default():
                self.dataGen = DATAPROCESS(train_data_path=pm.train_data_path,
                                          train_label_path=pm.train_label_path,
                                          test_data_path=pm.test_data_path,
                                          test_label_path=pm.test_label_path,
                                          word_embedings_path=pm.word_embedings_path,
                                          vocb_path=pm.vocb_path,
                                          batch_size=pm.batch_size
                                        )
                self.dataGen.load_wordebedding()
                self.dataGen.load_train_data()

                self.tag_nums = pm.tag_nums  # 标签数目
                self.hidden_nums = pm.hidden_nums  # bi-lstm的隐藏层单元数目
                self.sentence_len = self.dataGen.sentence_length # 句子长度,输入到网络的序列长度
                self.model_checkpoint_path = pm.model_checkpoint_path
                # print("pm.model_checkpoint_path", pm.model_checkpoint_path)
                self.dropout_keep_prob = pm.dropout_keep_prob
                self.model = BiLSTM_CRF(
                                        batch_size=batch_size,
                                        tag_nums=self.tag_nums,
                                        hidden_nums=self.hidden_nums,
                                        sentence_len=self.sentence_len,
                                        word_embeddings=self.dataGen.word_embeddings,
                                        device=device
                                        )
                self.saver = tf.train.Saver(max_to_keep=1)
                # 疑问 这里保存的是什么东西  是训练的模型吗？？？
                # tf.reset_default_graph()
                init = tf.initialize_all_variables()
                sess.run(init)

                self.load_NER_mode(sess)

    def nerApp(self, sess):
        # tf.reset_default_graph()
        with sess.as_default():
            with sess.graph.as_default():
                text = "application"
                while(text!="" and text!=" "):
                    text=input("请输入一句话：")
                    if text == "quit" or text=="" or text == " ":break
                    data_line,data_x,efficient_sequence_length=self.dataGen.handleInputData(text)
                    if debug:
                        print(np.array(data_x).shape)
                        print(data_x)
                        print(np.array(efficient_sequence_length).shape)
                    feed_dict={self.model.input_x:data_x,
                               self.model.sequence_lengths:efficient_sequence_length,
                               self.model.dropout_keep_prob:1.5}
                    predict_labels = sess.run([self.model.crf_labels],feed_dict)#predict_labels是三维的[1,1,25]，第1维包含了一个矩阵
                    lable_line =[]
                    if debug:
                        print(type(predict_labels))
                        print(predict_labels)
                        print(np.array(predict_labels).shape)
                    for idx in range(len(predict_labels[0])):
                        _label = predict_labels[0][idx].reshape(1,-1)
                        lable_line.append(list(_label[0]))
                    for idx in range(len(data_line)):
                        for each in range(efficient_sequence_length[idx]):
                            print("%s:%s"%(data_line[idx][each],lable_line[idx][each]), end="  ")
                        print('\n')


    def questionNer(self,sess,text):
        # tf.reset_default_graph()
        # init = tf.global_variables_initializer()
        # sess.run(init)
        if text== " ":
            print("文本为空，错误")
            return

        data_line,data_x,efficient_sequence_length=self.dataGen.handleInputData(text)
        # print("data_line,data_x,efficient_sequence_length", data_line, '\n', data_x, '\n', efficient_sequence_length)
        feed_dict = {self.model.input_x: data_x,
                     self.model.sequence_lengths: efficient_sequence_length,
                     self.model.dropout_keep_prob: 1.5
                     }

        predict_labels = sess.run([self.model.crf_labels],feed_dict)#predict_labels是三维的[1,1,25]，第1维包含了一个矩阵
        # print("self.model", self.model)
        # print("self.model.crf_labels", self.model.crf_labels)
        # print("predict_labels", predict_labels)
        lable_line =[]
        for idx in range(len(predict_labels[0])):
            _label = predict_labels[0][idx].reshape(1,-1)
            lable_line.append( list(_label[0]))
        return data_line, lable_line, efficient_sequence_length

    def train(self, session):
        # tf.reset_default_graph()
        # 创建一个默认会话
        with sess.as_default():
            with sess.graph.as_default():

                tf.summary.scalar('loss', self.model.loss)
                writer = tf.summary.FileWriter(self.model_checkpoint_path)

                #tf.global_variables_initializer.run()
                #init = tf.global_variables_initializer()
                #sess.run(tf.local_variables_initializer())
                #session.run(init)
                #sess.run(tf.global_variables_initializer())
                #sess.run(tf.local_variables_initializer())

                writer.add_graph(session.graph)
                for epoch in range(5):
                    print('Epoch:', epoch)
                    num_batchs = int((len(self.dataGen.train_data_raw) - 1) / pm.batch_size) + 1
                    train_batch = self.dataGen.batch_iter(self.dataGen.train_data_raw, self.dataGen.train_label_raw)
                    for x_batch, y_batch in train_batch:
                        x_batch, seq_leng_x = self.dataGen.pad_batch(x_batch)
                        y_batch, seq_leng_y = self.dataGen.pad_batch(y_batch)
                        feed_dict1 = self.model.feed_data(x_batch, y_batch, seq_leng_x, self.dropout_keep_prob)
                        _, global_step, pred_l, loss = session.run(
                            [self.model.optimizer, self.model.global_step, self.model.crf_labels,self.model.loss],
                            feed_dict=feed_dict1)
                        # writer.add_summary(loss, global_step=global_step)  # 写入文件
                        if global_step % 1 == 0:
                            predict_right_cnt, predict_cnt, real_cnt, accuracy = self.dataGen.evaluate(pred_l,
                                                                                                       y_batch,
                                                                                                       seq_leng_x)
                            print('global_step:', global_step, 'train_loss:', loss, "accuracy:", accuracy)

                        if global_step % 100 == 0:
                            test_loss,predict_lable,real_lable,seq_len = self.test(session)
                            predict_right_cnt,predict_cnt,real_cnt,accuracy = self.dataGen.evaluate(predict_lable,real_lable,seq_len)
                            print('global_step:', global_step, 'train_loss:', loss, 'test_loss:', test_loss, "predict_right_cnt:", predict_right_cnt,
                                  "predict_cnt:", predict_cnt, "real_cnt:", real_cnt, "accuracy:", accuracy)

                        if global_step % ( num_batchs//4) == 0:
                            print('Saving Model...')
                            save_path = os.path.join(self.model_checkpoint_path, "best_validation" + str(global_step))
                            self.saver.save(session, save_path=save_path, global_step=global_step)

                    pm.learning_rate *= pm.lr
                    #print("pm.learning_rate", pm.learning_rate)
                    # pm.lr = pm.lr
                    '''
                    while self.dataGen.train_batch_index<len(self.dataGen.train_batches):
                        x_batch,y_batch, seq_leng_x = self.dataGen.next_train_batch()
                        feed_dict1 = self.model.feed_data(x_batch, y_batch, seq_leng_x, self.dropout_keep_prob)

                        _, global_step, loss = session.run([self.model.optimizer, self.model.global_step, self.model.loss],
                                                                         feed_dict=feed_dict1)
                        #writer.add_summary(loss, global_step=global_step)  # 写入文件
                        if global_step % 3 == 0:
                            test_loss = self.model.test(session, self.dataGen)
                            print('global_step:', global_step, 'train_loss:', loss, 'test_loss:', test_loss)

                        if global_step % (4*num_batchs) == 0:
                            print('Saving Model...')
                            self.saver.save(session, save_path=self.model_checkpoint_path, global_step=global_step)
                    pm.learning_rate *= pm.lr
                    '''

    def test(self, sess ):
        # tf.reset_default_graph()
        datalen = len(self.dataGen.valid_data_raw)
        num_batch = int((datalen - 1) / pm.batch_size) + 1
        loss = 0
        batch_test = self.dataGen.batch_iter(self.dataGen.valid_data_raw,self.dataGen.valid_label_raw,pm.batch_size)
        #print("valid_data_raw:", self.dataGen.valid_data_raw[:10])
        #print("valid_label_raw:", self.dataGen.valid_label_raw[:10])

        y_pred_cls = []
        y_test_cls =[]
        y_length=[]
        for x_batch, y_batch in batch_test:
            x_batch, seq_leng_x = self.dataGen.pad_batch(x_batch)
            y_batch, seq_leng_y = self.dataGen.pad_batch(y_batch)

            feed_dict1 = self.model.feed_data(x_batch,y_batch,seq_leng_x,pm.dropout_keep_prob)
            pred_lable,temp_loss = sess.run([self.model.crf_labels,self.model.loss], feed_dict=feed_dict1)

            for i in range(len(pred_lable)):

                y_pred_cls.append(pred_lable[i])
                y_test_cls.append(y_batch[i])
                y_length.append(seq_leng_x[i])
            loss+=temp_loss
            #y_pred_cls[start_id:end_id] = sess.run([self.model.crf_labels], feed_dict=feed_dict1)
        return loss/num_batch,y_pred_cls,y_test_cls,y_length

    def test2(self,sess):
        # tf.reset_default_graph()

        totallen = len(self.dataGen.datas)
        for index in range(totallen):
            print(index)

            text = ''.join(self.dataGen.text_train[index][:-1].split(" "))

            lable = self.dataGen.text_label[index]
            data_line, data_x, efficient_sequence_length = self.dataGen.handleInputData(text)

            feed_dict = {self.model.input_x: data_x,
                         self.model.sequence_lengths: efficient_sequence_length,
                         self.model.dropout_keep_prob: 1.5}
            predict_labels= sess.run([self.model.crf_labels],
                                      feed_dict=feed_dict)  # predict_labels是三维的[1,1,25]，第1维包含了一个矩阵
            print("real_lable:",lable,"predict_lable",predict_labels)

    def load_NER_mode(self,sess):
        # tf.reset_default_graph()

        ckpt = tf.train.get_checkpoint_state(self.model_checkpoint_path)
        # print("ckpt", "\n", ckpt)
        #print("ckpt.all_model_checkpoint_paths", "\n", ckpt.all_model_checkpoint_paths)
        #print("ckpt.model_checkpoint_path", "\n", ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            #self.saver.restore(sess, tf.train.latest_checkpoint(self.model_checkpoint_path))
            logging.info("model loading successful")
        else:
            print("no checkpoint found.")


if __name__=="__main__":

    graph = tf.Graph()
    log_device_placement = False  # 是否打印设备分配日志
    allow_soft_placement = False  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session_conf = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)

    sess = tf.Session(graph=graph, config=session_conf)

    app = nerAppication(sess)
    app.train(sess)
    # app.test2(sess)

    # test_loss,predict_lable,real_lable,seq_len = app.test(sess)
    #predict_right_cnt, predict_cnt, real_cnt, accuracy = app.dataGen.evaluate(predict_lable, real_lable, seq_len)
    #print( 'test_loss:', test_loss, "predict_right_cnt:",
    #      predict_right_cnt,
    #      "predict_cnt:", predict_cnt, "real_cnt:", real_cnt, "accuracy:", accuracy)

    text="我发烧流鼻涕怎么办"
    while(text!="" and text!=" "):
        text=input("请输入一句话：")
        if text == "quit" or text=="" or text == " ":break
        data_line,lable_line,efficient_sequence_length = app.questionNer(sess,text)
        for idx in range(len(data_line)):
            for each in range(efficient_sequence_length[idx]):
                print("%s:%s"%(data_line[idx][each],lable_line[idx][each]),end="  ")
            print('\n')

