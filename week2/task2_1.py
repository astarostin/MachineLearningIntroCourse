import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
import numpy as np


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


data = np.array(pd.read_csv('wine.data', header=None))
y = data[:, 0].astype(int)
X = np.delete(data, 0, 1)
kf = KFold(len(X), n_folds=5, shuffle=True, random_state=42)
k_scores = []
for k in xrange(1, 51):
	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
	k_scores.append((k, scores.mean()))

k_scores.sort(key=lambda t: t[1], reverse=True)
print k_scores
out('2_1_1.txt', str(k_scores[0][0]))
out('2_1_2.txt', '%.2f' % (k_scores[0][1]))

X = scale(X)
k_scores = []
for k in xrange(1, 51):
	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
	k_scores.append((k, scores.mean()))

k_scores.sort(key=lambda t: t[1], reverse=True)
print k_scores
out('2_1_3.txt', str(k_scores[0][0]))
out('2_1_4.txt', '%.2f' % (k_scores[0][1]))