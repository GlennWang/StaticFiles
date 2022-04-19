import cv2
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# 讀取video
videocap = cv2.VideoCapture("test_dataset.avi")
# 讀取標籤
label = np.float16(np.loadtxt("label.txt"))
'''
或者用open
with open('lable.txt', 'r') as f:
    label = []
    file = f.readlines()
    for line in file:
        label.append(float(line))
'''

# 存放幀
frame = []
# 存放計算幀數
framecount = 0
# 存放且計算幀
while (True):
    ret, image = videocap.read()
    if ret is False:
        break
    framecount = framecount + 1

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    frame.append(image)

# 存入np矩陣
data = np.array(frame).reshape(framecount, -1)

clf = KNeighborsClassifier()
# 分割
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.5)
# 訓練
clf.fit(x_train, y_train)
# 預測
predit = clf.predict(x_test)
# 結果
print(metrics.classification_report(y_test, predit))
# 誤判
confusion = metrics.confusion_matrix(y_test, predit)
# 準確率
accuracy = np.diag(confusion).sum()/confusion.sum()

print(confusion)
print(f"準確率:{accuracy}")
