"""



Note: The code is used to show the change trende via the whole training procession.
First: You need to mark all the loss of every iteration
Second: You need to write these data into a txt file with the format like:
......
iter loss
iter loss
......
Third: the path is the txt file path of your loss



"""

import matplotlib.pyplot as plt
# import matplotlib.colorbar
import matplotlib as mpl


def read_txt(path):
    with open(path, 'r',encoding="utf-8") as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    return splitlines


# Referenced from Tensorboard(a smooth_loss function:https://blog.csdn.net/charel_chen/article/details/80364841)
def smooth_loss(path, weight=0.85):
    iter = []
    loss = []
    data = read_txt(path)
    for value in data:
        iter.append(int(value[0]))
        loss.append(int(float(value[1])))
        # Note a str like '3.552' can not be changed to int type directly
        # You need to change it to float first, can then you can change the float type ton int type
    last = loss[0]
    smoothed = []
    for point in loss:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return iter, smoothed
def train_precision(path,weight=0.85):
    iter = []
    precision = []
    data = read_txt(path)
    for value in data:
        iter.append(int(value[0]))
        precision.append(int(float(value[1])))
        # Note a str like '3.552' can not be changed to int type directly
        # You need to change it to float first, can then you can change the float type ton int type
    last = precision[0]
    smoothed = []
    for point in precision:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return iter,smoothed



if __name__ == "__main__":
    loss_path = 'C:\\Users\\940818lcq\\Desktop\\文本分类结果展示\\(2,3,4)\\loss-80000.txt'
    loss_path1 = "C:\\Users\\940818lcq\\Desktop\\文本分类结果展示\\(3,4,5)\\b-loss.txt"
    loss_path2 = "C:\\Users\\940818lcq\\Desktop\\文本分类结果展示\\(4,5,6)\\c-loss.txt"
    # accuracy_path = "C:\\Users\\940818lcq\\Desktop\\train-precision.txt"
    # precision_path = "C:\\Users\\940818lcq\\Desktop\\train_precision.txt"
    #
    # test_precision_path = "C:\\Users\\940818lcq\\Desktop\\test_precision.txt"

    # precision = []
    # loss = []
    # iter = []
    # iter, loss = smooth_loss(loss_path)
    # accuracy = []
    # iter = []
    iter, loss = smooth_loss(loss_path)
    iter1,loss1 = smooth_loss(loss_path1)
    iter2, loss2 = smooth_loss(loss_path2)
    # iter,accuracy =smooth_loss(accuracy_path)
    # plt.plot(iter,accuracy,lw=2,color="b")
    plt.plot(iter, loss, linewidth=2,color="c",ls="-",label="filter_sizes=2,3,4",marker=",")
    plt.plot(iter1,loss1,linewidth=2,color="b",ls="-.",label="filter_sizes=3,4,5",marker=",")
    plt.plot(iter2,loss2,linewidth=2,color="r",ls=":",label="filter_sizes=4,5,6",marker=",")
    plt.legend()
    plt.title("Model loss", fontsize=14)
    plt.xlabel("iter", fontsize=11)
    plt.ylabel("loss", fontsize=11)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig('C:\\Users\\940818lcq\\Desktop\\train-precision.png',dpi=800)
    plt.show()

import re
import csv
# f = open("C:\\Users\\940818lcq\\Desktop\\基于字向量BiLSTM模型结果.txt","r",encoding="utf-8")
# # b = open("C:\\Users\\940818lcq\\Desktop\\BiLSTM-character.txt","w",encoding="utf-8")
# c = []
# for line in f:
#     data = line.split(",")
#     if len(data) > 1 and "Val" not in data[1]:
#         a = re.findall('(?<=Loss:).*$', data[1])
#         c.append(a[0])
# f.close()
#
# with open('C:\\Users\\940818lcq\\Desktop\\BiLSTM-character-result3.csv', "w",newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     for line in c:
#             writer.writerow([line])
# f = open("C:\\Users\\940818lcq\\Desktop\\基于字向量的BiLSTM-CRF模型结果.txt","r",encoding="utf-8")
# lines = f.readlines()
# with open("C:\\Users\\940818lcq\\Desktop\\b.csv","w",newline="")as f:
#     writer = csv.writer(f)
#     for line in lines:
#         if 'Val' not in line and len(line) > 10:
#             print(line)
#             a = re.findall('(?<=Loss:).*$',line)
#             writer.writerow(a)



#     data = line.split(",")
#     if len(data) > 1 and "Val" not in data[1]:
#         a = re.findall('(?<=Loss:).*$', data[1])
#         c.append(a[0])
# f.close()

# with open('C:\\Users\\940818lcq\\Desktop\\BiLSTM-character-result3.csv', "w",newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     for line in c:
#             writer.writerow([line])
'''
这是我自己模型跑到结果数据整理代码
'''
# f = open("C:\\Users\\940818lcq\\Desktop\\基于词向量的BiLSTM-CRF模型结果2.txt","r",encoding="utf-8")
# lines = f.readlines()
# a = []
# for line in lines:
#     if len(line) < 75 and "Epoch" not in line:
#         a.append(line)
# train_loss = []
# train_accuracy = []
# for i in a:
#     a = re.findall('.*train_loss:(.*)accuracy:',i)
#     b = re.findall('(?<=accuracy:).*$',i)
#     print(b)
#     if a != 0:
#         train_loss.append(a[0].strip())
#     train_accuracy.append(b[0].strip())
# rows = zip(train_loss,train_accuracy)
# with open("C:\\Users\\940818lcq\\Desktop\\a.csv","w",newline="")as f:
#     for row in rows:
#         writer = csv.writer(f)
#         writer.writerow(row)

'''
# 两个准确率获取
'''
# # train_loss = []
# train_accuracy = []
# import re,csv
# f = open("C:\\Users\\940818lcq\\Desktop\\文本分类结果展示\\(2,3,4)\\a-80000.txt","r",encoding="utf-8")
# lines = f.readlines()
# print(len(lines))
# for line in lines:
#         # a = re.findall(".*loss:(.*)accuracy:",line)
#         # b = re.findall("(?<=accuracy:).*$",line)
#         b = re.findall(".*accuracy:(.*)train_precision:",line)
#         print(b)
#         # train_loss.append(a[0].strip())
#         train_accuracy.append(b[0].strip())
#
#
# rows = zip(train_accuracy)
#
# with open("C:\\Users\\940818lcq\\Desktop\\train-precision.csv","w",newline="",encoding="utf-8")as file:
#     for row in rows:
#         writer = csv.writer(file)
#         writer.writerow(row)
# f.close()
# file.close()



