import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


data_train = np.array(pd.read_csv('perceptron-train.csv', header=None))
data_test = np.array(pd.read_csv('perceptron-test.csv', header=None))
y_train = data_train[:, 0].astype(int)
y_test = data_test[:, 0].astype(int)
X_train = np.delete(data_train, 0, 1)
X_test = np.delete(data_test, 0, 1)
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc1 = accuracy_score(y_test, predictions, normalize=False) / float(len(y_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc2 = accuracy_score(y_test, predictions, normalize=False) / float(len(y_test))

print acc1, acc2

out('f2_3.txt', '%.2f' % (acc2 - acc1))