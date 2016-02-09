import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydot
import pyparsing


data = pd.read_csv('titanic.csv',
				   usecols=['Pclass', 'Sex', 'Age', 'Fare', 'Survived'], \
				   converters={'Sex': lambda s: 0 if s == 'male' else 1})

data = data.dropna()
X = data[:]
del X['Survived']
y = data['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
importances = sorted(zip(X.columns, clf.feature_importances_), key=lambda t: t[1], reverse=True)
res = '%s %s' % (importances[0][0], importances[1][0])
print res
# write to the file
f = open('f1_2.txt', 'w')
f.write(res)
f.close()
# draw the tree
dot_data = StringIO()
export_graphviz(clf, feature_names=data.columns.values, class_names=['DEAD', 'ALIVE'], filled=True, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("titanic_tree.pdf")
