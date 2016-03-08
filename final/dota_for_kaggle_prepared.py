import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# read training data
data = pd.read_csv('train_prepared.csv', index_col='match_id')

# optional line: pick random N rows from data for quicker testing
# data = data.ix[np.random.choice(data.index, 10000, replace=False)]

# separate target variable and training features
Y = data['radiant_win']
del data['radiant_win']

# Learning LogisticRegression with cross validation and various regularization coefficient values.
# Analyzing AUC_ROC scores.
start_time = datetime.datetime.now()
kf = KFold(len(data), n_folds=5, shuffle=True, random_state=241)
# estimators_grid = {'C': [10 ** k for k in np.arange(-3, 0.1, 0.1)]}
# estimators_grid = {'C': np.arange(0.075, 0.085, 0.002), 'penalty': ['l1', 'l2']}
# lr = LogisticRegression(random_state=241)
c_param = 0.0794
lr = LogisticRegression(C=c_param, random_state=241)
scores = cross_val_score(lr, data, Y, cv=kf, scoring='roc_auc')
# gs = GridSearchCV(lr, estimators_grid, scoring='roc_auc', cv=kf, verbose=1)
# gs.fit(data, Y)
elapsed = datetime.datetime.now() - start_time
# results = [(x.parameters['C'], x.parameters['penalty'], x.mean_validation_score) for x in gs.grid_scores_]
# for result in results:
# 	print '%.4f, %s - %.8f' % (result[0], result[1], result[2])
# print 'Best parameter C = %.4f, penalty = %s, best score = %.8f' % (gs.best_params_['C'], gs.best_params_['penalty'] , gs.best_score_)
print 'time = %s, score = %.8f' % (str(elapsed), scores.mean())


# Make predictions for test data
data_test = pd.read_csv('test_prepared.csv', index_col='match_id')

# Apply LogisticRegression with the best found value for C parameter
lr = LogisticRegression(C=c_param)

# Learn on train data
lr.fit(data, Y)

# Make predictions for test data
predictions = lr.predict_proba(data_test)[:, 1]
print 'Predictions: min = %.4f, max = %.4f' % (np.min(predictions), np.max(predictions))

# Create a dataframe with predictions and write it to CSV file
predictions_df = pd.DataFrame(data=predictions, index=data_test.index, columns=['radiant_win'])
predictions_df.to_csv('predictions.csv', sep=',')