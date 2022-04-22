dic1 = {
    "0":0,
    "1":0,
    "2":0,
    "3":0,
    "4":0,
    "5":0,
    "6":0,
    "7":0,
    "8":0,
}
f = open("D:\\PycharmProjects\\chatbot_KnowledgeGrapg_modify\\data_ai\\classifyData\\train_data.txt",encoding="utf-8")
content = f.readlines()
for i in content:
    new_data = i.split()
    if new_data[0] in dic1:
        dic1[new_data[0]] += 1
print(dic1)

