import json
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')  # 忽略警告

import logging
# 这个库的作用:提供记录日志功能
import os.path
import sys
import multiprocessing
import numpy as np

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    # print open('/Users/sy/Desktop/pyRoot/wiki_zh_vec/cmd.txt').readlines()
    # sys.exit()
    # sys.argv[0]返回的是当前.py文件的路径
    program = os.path.basename(sys.argv[0])
    # 这里program是word2vec.py
    logger = logging.getLogger(program)
    # 设置日志记录器

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    # 实现打印日志的基础配置  format:指定输出的格式和内容   level:设置日志级别
    logger.info("running %s" % ' '.join(sys.argv))

    # inp为输入语料, outp1 为输出模型, outp2为原始c版本word2vec的vector格式的模型

    # 实体识别词向量

    inp = './data_ai/nerData/train_cutword_data.txt'
    outp = './data_ai/cbowData/'



    # ner词向量
    # model = Word2Vec(LineSentence(inp), size=128, window=5, min_count=5,
    #                  workers=multiprocessing.cpu_count())
    # ind = [];
    #
    # for index in range(len(model.wv.index2word)):
    #     ind.append(index)
    # di = dict(zip(ind, model.wv.index2word))
    # with open(outp + "ner2.vab", 'w') as file_obj:
    #     json.dump(di, file_obj)
    #
    #
    # model.save(outp+"ner_model2.model")


    #意图识别词向量

    inp = './data_ai/classifyData/train_cutword_data.txt'
    outp = './data_ai/cbowData/'


    model = Word2Vec(LineSentence(inp), size=128, window=5, min_count=5,workers=multiprocessing.cpu_count())
    ind = [];

    for index in range(len(model.wv.index2word)):
        ind.append(index)
    di = dict(zip(ind, model.wv.index2word))
    with open(outp + "classify1.vab", 'w') as file_obj:
        json.dump(di, file_obj)

    # 保存模型
    model.save(outp + "classify_model1.model")
#
# #     命名实体识别字向量训练
#
# inp = './data_ai/nerData/train_data采用的字向量.txt'
# outp = './data_ai/cbowData/'



# # ner词向量
# model = Word2Vec(LineSentence(inp), size=200, window=5, min_count=5,
#                  workers=multiprocessing.cpu_count())
# ind = [];
#
# for index in range(len(model.wv.index2word)):
#     ind.append(index)
# di = dict(zip(ind, model.wv.index2word))
# with open(outp + "ner1.vab", 'w') as file_obj:
#     json.dump(di, file_obj)
#
#
# model.save(outp+"ner_model1.model")
#

