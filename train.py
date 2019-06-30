#coding:utf-8
#1.导入所需的库
import csv
import numpy as np
from dasai.vector import WordVector
from sklearn.svm import SVC
import pickle

#2.创建字向量计算对象
wordVector = WordVector()
#准备features 和label 集合
data = []
label  = []
#3.定义用于计算队员数和指导老师数目的函数
def authorNum(line):
    res = 0
    for i in range(7,12):
        if line[i] != '':
            res += 1
    return res
def teacherNum(line):
    res = 0
    for i in range(12,14):
        if line[i] != '':
            res += 1
    return res

#4.读入并处理csv文件
f1 = csv.reader(open("2018.csv","r",encoding="UTF-8"))
f2 = csv.reader(open("2017.csv","r",encoding="UTF-8"))
f3 = csv.reader(open("2017.csv","r",encoding="UTF-8"))

for line in f1:
    label.append([int(line[0])])
    data_line = np.hstack([wordVector.getWordVector(line[4]),wordVector.getWordVector(line[5]),[authorNum(line)],[teacherNum(line)]])
    data.append(data_line)
for line in f2:
    label.append([int(line[0])])
    data_line = np.hstack([wordVector.getWordVector(line[4]),wordVector.getWordVector(line[5]),[authorNum(line)],[teacherNum(line)]])
    data.append(data_line)
for line in f3:
    label.append([int(line[0])])
    data_line = np.hstack([wordVector.getWordVector(line[4]),wordVector.getWordVector(line[5]),[authorNum(line)],[teacherNum(line)]])
    data.append(data_line)
data = np.array(data)
# 5.定义SVM模型
svc = SVC(C=1.5,tol=1e-5,class_weight="balanced")
svc.fit(data,label)
# 6.存储模型，方便下次使用
f = open("SVC.model","wb")
pickle.dump(svc,f)
f.close()

# 7。预测我们作品的情况
test = np.array([np.hstack([wordVector.getWordVector("法规宝"), wordVector.getWordVector("上海财经大学"),[3],[1]])])
print(svc.predict(test))
