import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


def make_step(w1, w2, k, C, X, y):
	sum1 = 0
	sum2 = 0
	for i in xrange(0, len(X)):
		linear_combination = w1 * X[i][0] + w2 * X[i][1]
		in_parenthesis = 1 - 1 / (1 + np.exp(-y[i] * linear_combination))
		sum1 += y[i] * X[i][0] * in_parenthesis
		sum2 += y[i] * X[i][1] * in_parenthesis
	w1 = w1 + (k / len(X)) * sum1 - k * C * w1
	w2 = w2 + (k / len(X)) * sum2 - k * C * w2
	return w1, w2


def dist(w1, w2, w1_prev, w2_prev):
	return np.sqrt((w1 - w1_prev)**2 + (w2 - w2_prev)**2)


def score_item(w1, w2, x1, x2):
	return 1 / (1 + np.exp(-w1 * x1 - w2 * x2))


def score(w1, w2, X):
	return [score_item(w1, w2, x[0], x[1]) for x in X]


def gradient_descent(X, y, c):
	k = 0.1
	max_iter = 10000
	eps = 1e-5
	(w1_prev, w2_prev) = (0, 0)
	cur_iter = 0
	cur_dist = 1

	while cur_dist > eps and cur_iter < max_iter:
		w1, w2 = make_step(w1_prev, w2_prev, k, c, X, y)
		cur_dist = dist(w1, w2, w1_prev, w2_prev)
		cur_iter += 1
		w1_prev, w2_prev = w1, w2
	return w1, w2, cur_iter


data = np.array(pd.read_csv('data-logistic.csv', header=None))
y = data[:, 0].astype(int)
X = np.delete(data, 0, 1)

w1, w2, cur_iter = gradient_descent(X, y, 0)
print 'Finished after %s iterations with w1 = %.4f w2 = %.4f' % (cur_iter, w1, w2)
score1 = roc_auc_score(y, score(w1, w2, X))

w1, w2, cur_iter = gradient_descent(X, y, 10)
print 'Finished after %s iterations with w1 = %.4f w2 = %.4f' % (cur_iter, w1, w2)
score2 = roc_auc_score(y, score(w1, w2, X))

res = '%.3f %.3f' % (score1, score2)
print res
out('3_3.txt', res)