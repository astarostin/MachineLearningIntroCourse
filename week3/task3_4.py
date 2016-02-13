import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


def out(filename, s):
	f = open(filename, 'w')
	f.write(s)
	f.close()


data = pd.read_csv('classification.csv', header=0)
dt = data['true']
dp = data['pred']
tp = data[(dt == 1) & (dp == 1)]
tn = data[(dt == 0) & (dp == 0)]
fp = data[(dt == 0) & (dp == 1)]
fn = data[(dt == 1) & (dp == 0)]

res = '%s %s %s %s' % (len(tp.index), len(fp.index), len(fn.index), len(tn.index))
print res
out('3_4.txt', res)

res = '%.2f %.2f %.2f %.2f' % (accuracy_score(dt, dp), precision_score(dt, dp), recall_score(dt, dp), f1_score(dt, dp))
print res
out('3_5.txt', res)

data = pd.read_csv('scores.csv', header=0)
roc_auc_data = {'score_logreg': roc_auc_score(data['true'], data['score_logreg']),
				'score_svm': roc_auc_score(data['true'], data['score_svm']),
				'score_knn': roc_auc_score(data['true'], data['score_knn']),
				'score_tree': roc_auc_score(data['true'], data['score_tree'])}
res = sorted(roc_auc_data.keys(), key=lambda k: roc_auc_data[k], reverse=True)[0]
print res
out('3_6.txt', res)


def get_max_precision((precision, recall, thr), recall_min=0.7):
	i = 0
	res = 0
	while recall[i] > recall_min:
		if precision[i] > res:
			res = precision[i]
		i += 1
	return res


pr_auc_data = {'score_logreg': get_max_precision(precision_recall_curve(data['true'], data['score_logreg'])),
			   'score_svm': get_max_precision(precision_recall_curve(data['true'], data['score_svm'])),
			   'score_knn': get_max_precision(precision_recall_curve(data['true'], data['score_knn'])),
			   'score_tree': get_max_precision(precision_recall_curve(data['true'], data['score_tree']))}
res = sorted(pr_auc_data.keys(), key=lambda k: pr_auc_data[k], reverse=True)[0]
print res
out('3_7.txt', res)
