import pickle
import numpy as np
import csv
from dasai.vector import WordVector

wordVector = WordVector()
svc = pickle.load(open("SVC.model","rb"))

f1 = csv.reader(open("2017.csv", "r", encoding="UTF-8"))
data = []
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
c = 0
for line in f1:
    data_line = np.hstack([wordVector.getWordVector(line[4]), wordVector.getWordVector(line[5]), [authorNum(line)],
                           [teacherNum(line)]])
    data.append(data_line)
    c += 1
    if c > 10:
        break

data = np.array(data)
print(svc.predict(data))
while True:
    a = input("作品")
    b = input("学校")
    test = np.array([np.hstack([wordVector.getWordVector(a), wordVector.getWordVector(b),[3],[1]])])
    print(svc.predict(test))