from sklearn import datasets, grid_search
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
#print len(newsgroups.data)
y = newsgroups.target

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(newsgroups.data)
#print X._shape
names = tfidf.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

best_c = gs.best_params_['C']

clf = SVC(kernel='linear', C=best_c, random_state=241)
clf.fit(X, y)
coefs = clf.coef_.toarray()[0]
major_coefs_indices = sorted(range(coefs.shape[0]), key=lambda k: abs(coefs[k]), reverse=True)[:10]
major_words = [names[k] for k in major_coefs_indices]
major_words.sort()
res = ' '.join(s for s in major_words)
print res
out('3_2.txt', res)


