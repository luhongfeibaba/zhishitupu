from gensim.models import word2vec
import json


def main():
    # num_features = 300  # Word vector dimensionality  词向量维度
    num_features = 200
    # min_word_count = 10  # Minimum word count
    min_word_count = 5
    num_workers = 16  # Number of threads to run in parallel  并行运行的线程数
    # context = 10  # Context window size  上下文窗口大小
    context = 5
    downsampling = 1e-3  # Downsample setting for frequent words  常用单词的下采设置
    sentences = word2vec.Text8Corpus("./data_ai/nerData/new_train_cutword_data1.txt")

    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sg=1, sample=downsampling)
    ind = []
    for index in range(len(model.wv.index2word)):
        ind.append(index)
    di = dict(zip(ind, model.wv.index2word))
    with open("./data_ai/cbowData/new_ner.vab", 'w') as file_obj:
        json.dump(di, file_obj)
    model.init_sims(replace=True)
    # 保存模型，供日後使用
    model.save("./data_ai/cbowData/new_ner_model.model")

    # 可以在加载模型之后使用另外的句子来进一步训练模型
    # model = gensim.models.Word2Vec.load('/tmp/mymodel')
    # model.train(more_sentences)


if __name__ == "__main__":
    main()


                     
'''
    inp = './data_ai/nerData/train_cutword_data.txt'
    outp = './data_ai/cbowData/'



    # ner词向量
    model = Word2Vec(LineSentence(inp), size=200, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
    ind = [];

    for index in range(len(model.wv.index2word)):
        ind.append(index)
    di = dict(zip(ind, model.wv.index2word))
    with open(outp + "ner.vab", 'w') as file_obj:
        json.dump(di, file_obj)


    model.save(outp+"ner_model.model")
'''

'''
数据合并
'''
'''
f = open("./data_ai/nerData/train_cutword_data.txt","r",encoding="utf-8")
b = open("./data_ai/nerData/a.txt","w",encoding="utf-8")
lines = f.readlines()
with open("./data_ai/nerData/a.txt","w",encoding="utf-8")as f:
    for line in lines:
        c = line.split(" ")
        d = "".join(c)
        f.write(d)
f.close()
'''

'''
重新分词
'''
'''
# -*- coding: utf-8 -*-

import jieba

# 加载自己的自己的医学词典
# jieba.load_userdict("./dict/disease.txt")
# jieba.load_userdict("./dict/check.txt")
# jieba.load_userdict("./dict/drug.txt")
# jieba.load_userdict("./dict/symptom.txt")
# jieba.load_userdict("./dict/food.txt")


def main():
    with open('./data_ai/nerData/a.txt', 'r', encoding='utf-8') as content:
        for line in content:
            seg_list = jieba.cut(line)
            with open('./data_ai/nerData/new_train_cutword_data1.txt', 'a', encoding='utf-8') as output:
                output.write(' '.join(seg_list))


if __name__ == '__main__':
    main()
'''

'''
对比分词是否正确
'''

# f = open('./data_ai/nerData/new_train_cutword_data1.txt', 'r', encoding='utf-8').readlines()
# f1 = open('./data_ai/nerData/train_cutword_data.txt', 'r', encoding='utf-8').readlines()
# count = 0
# print(f[1] == f1[1])
# print(len(f[1]))
# print(len(f1[1]))
# for i in range(len(f)):
#     if f[i] == f1[i]:
#         print(f[i])

# 加入分词词典后,每一个分词结果都不一样





