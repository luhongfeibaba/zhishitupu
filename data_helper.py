import json
import re


# filepath = "D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\CMID-master\\CMID.json"
# trainFile_path = "D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train2.txt"
# file = open(trainFile_path,"w",encoding="utf-8")
# for data in open(filepath,encoding='UTF-8') :
#     a = eval(data)
#     data_dict = a[0]
#     try:
#         if data_dict['label_36class'][0] == "临床表现":
#             result = data_dict["originalText"]
#             file.write(result)
#             file.write("\n")
#             # result = " ".join(data_dict['seg_result'])
#             # new_data = "0 "+result
#             # file.write(new_data)
#             # file.write("\n")
#         if data_dict['label_36class'][0] == "治疗方法":
#             result = data_dict["originalText"]
#             file.write(result)
#             file.write("\n")
#             # result = " ".join(data_dict['seg_result'])
#             # new_data = "8 "+result
#             # file.write(new_data)
#             # file.write("\n")
#         if data_dict['label_36class'][0] == "预防":
#             result = data_dict["originalText"]
#             file.write(result)
#             file.write("\n")
#             # result = " ".join(data_dict['seg_result'])
#             # new_data = "6 "+result
#             # file.write(new_data)
#             # file.write("\n")
#         if data_dict['label_36class'][0] == "定义":
#             result = data_dict["originalText"]
#             file.write(result)
#             file.write("\n")
#             # result = " ".join(data_dict['seg_result'])
#             # new_data = "9 "+result
#             # file.write(new_data)
#             # file.write("\n")
#         if data_dict['label_36class'][0] == "病因":
#             result = data_dict["originalText"]
#             file.write(result)
#             file.write("\n")
#             # result = " ".join(data_dict['seg_result'])
#             # new_data = "10 "+result
#             # file.write(new_data)
#             # file.write("\n")
#     except:
#         pass
#     continue
# file.close()
#
#先获取每一个问句的分类标签

# a = "D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train3.txt"
# c = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\label.txt","w",encoding="utf-8")
# with open(a,"r",encoding="utf-8")as f:
#     data = f.readlines()
#     for i in data:
#         b = i.split(" ")
#         c.write(b[0])
#         c.write("\n")
# c.close()

# 下面是对问句进行分词和去停用词并加载自己构建的字典
# import jieba
# filepath = "D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train2.txt"
# with open(filepath,"r",encoding="utf-8")as f:
#     data = f.readlines()
#     for i in data:
#         print(i)
#
import jieba
import jieba.posseg as pseg
# from jieba import analyse

# 加载停用词库
# def stopwordslist():
#     stopwords = [line.strip() for line in open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\stop_words.txt","r",encoding="utf-8").readlines()]
#     return stopwords
# stop = [line.strip() for line in open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\stop_words.txt","r",encoding="utf-8").readlines()]
# 导入自定义词典

# jieba.load_userdict("C:\\Users\\940818lcq\\Desktop\\dict1\\dict.txt")
# 分词代码
# f = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train4.txt","r",encoding="utf-8")
# f1 = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train5.txt","w",encoding="utf-8")
# contences = f.readlines()
# for sentence in contences:
#     segs = jieba.cut(sentence,cut_all=False)
#     final = ""
#     for seg in segs:
#         final +=' '+seg
#     f1.write(final)
# f1.close()

#下面将对应问句标签与标注在对应问句中
# a = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train5.txt","r",encoding="utf-8")
# b = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\label.txt","r",encoding="utf-8")
# c = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train6.txt","w",encoding="utf-8")
# a_data = a.readlines()
# b_data = b.readlines()
# for i in range(len(a_data)):
#     #需要对标签数据进行换行符的去除
#     final = b_data[i].strip()+a_data[i]
#     print(final)
#     c.write(final)
# c.close()
# a.close()

# b.close()

# 重现按照比例进行数据的划分

# import re
# text = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\total_data1.txt","r",encoding="utf-8").read()
# b = re.split('\n',text)
# n = 0
# for i in b:
#     n += 1
# print(n)
# m = 0
# for i in b:
#     m += 1
#     if m <= n*7/10:
#         with open('D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train1\\%s.txt'%m,'w',encoding="utf-8")as f:
#             f.write(i)
#             print(m)
#
#     else:
#         with open('D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\text1\\%s.txt'%m,'w',encoding='utf-8')as f:
#             f.write(i)
#             print(m)

# 合并数据集
# import os
# dirPath = 'D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train'
# files = os.listdir(dirPath)
# res = ""
# i = 0
# for file in files:
#     # if file.endwith('.txt'):
#     i += 1
#     title = "第%s章 %s"%(i,int(file[0:len(file)])-4)
#     with open('D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train\\' + file,'r',encoding="utf-8")as f:
#         content = file.read()
#         file.close()
#
#     append = '\n%s\n\n%s'%(title,content)
#
#     res += append
# with open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\zong.txt",'w',encoding="utf-8")as f:
#     f.write(res)
#     f.close()
# print(len(res))
import os
filedir = "D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\classifyData\\text1"
filenames = os.listdir(filedir)
f = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\classifyData\\result1.txt",'w')
for filename in filenames:
    filepath = filedir + '/' + filename
    for line in open(filepath,'r',encoding="utf-8"):
        f.write(line)
        f.write('\n')
f.close()