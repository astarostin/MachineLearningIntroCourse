import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


data = pd.read_csv('gbm-data.csv', header=0).values

x_train, x_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0], test_size=0.8, random_state=241)
# for lr in [1, 0.5, 0.3, 0.2, 0.1]:
for lr in [0.2]:
	print '############## RATE %s ##########' % lr
	clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=lr)
	clf.fit(x_train, y_train)
	train_score = []
	test_score = []
	for i, y_predicted in enumerate(clf.staged_decision_function(x_train)):
		train_score.append(log_loss(y_train, 1 / (1 + np.exp(-y_predicted))))
	for i, y_predicted in enumerate(clf.staged_decision_function(x_test)):
		test_score.append(log_loss(y_test, 1 / (1 + np.exp(-y_predicted))))
	plt.figure()
	plt.plot(test_score, 'g', linewidth=2)
	plt.plot(train_score, 'r', linewidth=2)
	plt.legend(['test', 'train'])
	#plt.show()
	n_iter = np.argmin(np.array(test_score))
	best = np.amin(np.array(test_score))
	res = '%.2f %d' % (best, n_iter)
	print res
	out('5_3.txt', res)

	clf2 = RandomForestClassifier(n_estimators=n_iter, random_state=241)
	clf2.fit(x_train, y_train)
	y_pred_test = clf2.predict_proba(x_test)[:, 1]
	test_loss = log_loss(y_test, 1 / (1 + np.exp(-y_pred_test)))
	res = '%.2f' % test_loss # 0.54
	print res
	out('5_4.txt', res)

out('5_2.txt', 'overfitting')


