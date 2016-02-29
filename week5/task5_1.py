from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


data = pd.read_csv('abalone.csv', header=0)
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))


y = data['Rings'].astype(float)
del data['Rings']

kf = KFold(len(data), n_folds=5, random_state=1, shuffle=True)
grid = {'n_estimators': np.arange(1, 51)}
rf = RandomForestRegressor(random_state=1)
gs = GridSearchCV(rf, grid, scoring='r2', cv=kf)
gs.fit(data, y)
res = str(next(x.parameters['n_estimators'] for x in gs.grid_scores_ if x.mean_validation_score > 0.52))
print res
out('5_1.txt', res)
