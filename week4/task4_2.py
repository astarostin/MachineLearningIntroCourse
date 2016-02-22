import pandas as pd
from sklearn.decomposition import PCA
from numpy import corrcoef
from math import fabs


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


data = pd.read_csv('close_prices.csv', header=0)
data2 = pd.read_csv('djia_index.csv', header=0)
prices = data.iloc[:, 1:].copy()
indices = data2.iloc[:, 1].copy()

pca = PCA(n_components=10)
pca.fit(prices)
s = 0
cnt = 0
for c in sorted(pca.explained_variance_ratio_, reverse=True):
	if s < 0.9:
		s += c
		cnt += 1
res = str(cnt)
print res
out("4_2.txt", res)

comps = pca.transform(prices)
res = '%.2f' % corrcoef(comps[:, 0], indices)[0, 1]
print res
out("4_3.txt", res)

max_influence_index = sorted(xrange(len(pca.components_[0])), key=lambda x: fabs(pca.components_[0][x]), reverse=True)[0]
res = prices.columns.values[max_influence_index]
print res
out("4_4.txt", res)
