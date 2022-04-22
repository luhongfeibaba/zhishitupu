import csv
import matplotlib.pyplot as plt

# file = "C:\\Users\\940818lcq\\Desktop\\data4.csv"

def a(path="C:\\Users\\940818lcq\\Desktop\\data4.csv"):
    csvData = open(path,'r')
    readcsv = csv.reader(csvData)
    loss = []
    acc = []
    for row in readcsv:
        loss.append(row[0])
        acc.append(row[1])
    csvData.close()
    loss.pop(0)
    acc.pop(0)
    return loss,acc
loss,acc = a()
loss = loss[:]
epoch = [i for i in range(1,13)]
plt.plot(epoch,acc,color="r",label="acc")
plt.plot(epoch,loss,color=(0,0,0),label="loss")

plt.xlabel("epoch")
plt.ylabel("y label")
plt.title("chart")
plt.legend()
plt.savefig("C:\\Users\\940818lcq\\Desktop\\result.jpg")
plt.show()