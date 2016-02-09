import pandas
from collections import Counter
import re


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
#print data.head()

res = '%s %s' % (len(data[data['Sex'] == 'male'].index), len(data[data['Sex'] == 'female'].index))
print res
out('f1.txt', res)

total = len(data)
res = '%.2f' % (len(data[data['Survived'] == 1].index) / float(total) * 100)
print res
out('f2.txt', res)

res = '%.2f' % (len(data[data['Pclass'] == 1].index) / float(total) * 100)
print res
out('f3.txt', res)

res = '%.2f %.2f' % (data['Age'].mean(), data['Age'].median())
print res
out('f4.txt', res)

res = '%.2f' % data.corr()['SibSp']['Parch']
print res
out('f5.txt', res)

cnt = Counter()
for line in data[data['Sex'] == 'female']['Name']:
	cnt.update([x for x in re.split(r'[" ,.()]+', line)[2:] if x])
res = cnt.most_common(1)[0][0]
print res
out('f6.txt', res)
