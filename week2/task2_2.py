from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import numpy as np


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()

boston = load_boston()

X = boston.data
y = boston.target
X = scale(X)
p_scores = []
kf = KFold(len(X), n_folds=5, shuffle=True, random_state=42)
for p in np.linspace(1.0, 10.0, num=200):
	knr = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
	scores = cross_val_score(knr, X, y, cv=kf, scoring='mean_squared_error')
	p_scores.append((p, scores.mean()))

p_scores.sort(key=lambda t: t[1], reverse=True)
print '%.2f' % (p_scores[0][0])
out('2_2.txt', '%.2f' % (p_scores[0][0]))