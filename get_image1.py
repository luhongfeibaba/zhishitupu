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
    with open(path, 'r') as f:
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
    # accuracy_path = "C:\\Users\\940818lcq\\Desktop\\word-character-accuracy.txt"
    loss_path = "C:\\Users\\940818lcq\\Desktop\\word-loss.txt"



    iter, loss = smooth_loss(loss_path)
    # iter,accuracy = train_precision(accuracy_path)
    # precision = train_precision(precision_path)
    plt.plot(iter, loss, linewidth=2,color="b")
    plt.title("test-loss", fontsize=24)
    plt.xlabel("iters", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig('C:\\Users\\940818lcq\\Desktop\\character-BiLSTM-CRF.png')
    plt.show()
