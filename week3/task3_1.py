import numpy as np
import pandas as pd
from sklearn.svm import SVC


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


data = np.array(pd.read_csv('svm-data.csv', header=None))
y = data[:, 0].astype(int)
X = np.delete(data, 0, 1)

svc = SVC(C=100000, kernel='linear', random_state=241)
svc.fit(X, y)
res = ' '.join(str(s+1) for s in svc.support_)
print res
out('3_1.txt', res)
