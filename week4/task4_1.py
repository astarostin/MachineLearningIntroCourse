import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


data_train = pd.read_csv('salary-train.csv', header=0)
data_test = pd.read_csv('salary-test-mini.csv', header=0)
Y = data_train['SalaryNormalized']
del data_train['SalaryNormalized']
del data_test['SalaryNormalized']

data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_train['LocationNormalized'] = data_train['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_train['LocationNormalized'] = data_train['LocationNormalized'].str.lower()
data_train['ContractTime'] = data_train['ContractTime'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_train['ContractTime'] = data_train['ContractTime'].str.lower()

data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_test['FullDescription'] = data_test['FullDescription'].str.lower()
data_test['LocationNormalized'] = data_test['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_test['LocationNormalized'] = data_test['LocationNormalized'].str.lower()
data_test['ContractTime'] = data_test['ContractTime'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_test['ContractTime'] = data_test['ContractTime'].str.lower()

tfidf = TfidfVectorizer(min_df=5)
X1_train = tfidf.fit_transform(data_train['FullDescription'])
X1_test = tfidf.transform(data_test['FullDescription'])

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X2_train = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X2_test = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X = hstack([X1_train, X2_train])
X_test = hstack([X1_test, X2_test])

print X.shape
print X_test.shape

classifier = Ridge(alpha=1)
classifier.fit(X, Y)

Y_test = classifier.predict(X_test)
res = "%.2f %.2f" % (Y_test[0], Y_test[1])
print res
out('4_1.txt', res)
