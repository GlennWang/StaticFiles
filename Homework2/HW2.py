import imp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, mean_squared_error
"""
簡略但可能看不懂法
with open("iris_x.txt") as f:
    x = []
    for line in range(打開看裡面有幾行):
        x.append(np.array(f.readline().split(), dtype=float))
    data = np.array(x)
with open("iris_y.txt") as f:
    label = np.array(f.read().split(), dtype=int) 
"""
# 正常讀檔案思路
with open("iris_x.txt", 'r') as f:
    x = []
    file = f.readlines()
    for line in file:
        index = line.split()
        x.append([float(index[0]), float(index[1]),
                 float(index[2]), float(index[3])])
    data = np.array(x)

with open("iris_y.txt", 'r') as f:
    y = []
    file = f.readlines()
    for line in file:
        y.append(int(line))
    label = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(
    data, label, test_size=0.2, random_state=20220413)

clt = LinearRegression()

clt.fit(x_train, y_train)

mse = mean_squared_error(y_test, clt.predict(x_test))

print(f'MSE:\n{mse}')


class QuadraticDiscriminantAnalysis_():
    def ___init__(self):
        self.mu = np.array([])
        self.cov = np.array([])

    def fit(self, data_train, label_train):
        mu, cov = [], []
        for i in range(np.max(label_train)+1):
            pos = np.where(label_train == i)[0]
            tmp_data = data_train[pos, :]
            tmp_cov = np.cov(np.transpose(tmp_data))
            tmp_mu = np.mean(tmp_data, axis=0)
            mu.append(tmp_mu)
            cov.append(tmp_cov)
        self.mu = np.array(mu)
        self.cov = np.array(cov)

    def predict(self, x_test):
        d_value = []
        for tmp_mu, tmp_cov in zip(self.mu, self.cov):
            d = len(tmp_mu)
            zero_center_data = x_test - tmp_mu
            tmp = np.dot(zero_center_data.transpose(), np.linalg.inv(tmp_cov))
            tmp = -0.5*np.dot(tmp, zero_center_data)
            tmp1 = (2 * np.pi)**(-d/2) * np.linalg.det(tmp_cov)**(-0.5)
            tmp = tmp1 * np.exp(tmp)
            d_value.append(tmp)
        d_value = np.array(d_value)
        return np.argmax(d_value), d_value


qda_ = QuadraticDiscriminantAnalysis_()

qda_.fit(x_train, y_train)
predict_ = []
for i in range(len(x_test)):
    pred, p = qda_.predict(x_test[i])
    predict_.append(pred)
confusion_ = confusion_matrix(y_test, predict_)
accuracy_ = np.diag(confusion_).sum()/confusion_.sum()

print(f'course confusion:\n{confusion_}')
print(f'cource accuracy:\n{accuracy_}')

qda = QuadraticDiscriminantAnalysis()

qda.fit(x_train, y_train)

predict = qda.predict(x_test)

confusion = confusion_matrix(y_test, predict)
accuracy = qda.score(x_test, y_test)

print(f'sklearn confusion:\n{confusion}')
print(f'sklearn accuracy:\n{accuracy}')
