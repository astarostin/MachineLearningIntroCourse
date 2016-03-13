import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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
estimators_grid = {'n_estimators': np.arange(1000, 1006, 10)}

rf = RandomForestClassifier(max_depth=5, random_state=241)

#scores = cross_val_score(lr, data, Y, cv=kf, scoring='roc_auc')
gs = GridSearchCV(rf, estimators_grid, scoring='roc_auc', cv=kf, verbose=1)
gs.fit(data, Y)
elapsed = datetime.datetime.now() - start_time

print 'Best parameter n_estimators = %.4f best score = %.8f' % (gs.best_params_['n_estimators'], gs.best_score_)
#print 'time = %s, score = %.8f' % (str(elapsed), scores.mean())


# Make predictions for test data
data_test = pd.read_csv('test_prepared.csv', index_col='match_id')

# Apply LogisticRegression with the best found value for C parameter
rf = RandomForestClassifier(n_estimators=gs.best_params_['n_estimators'])

# Learn on train data
rf.fit(data, Y)

# Make predictions for test data
predictions = rf.predict_proba(data_test)[:, 1]
print 'Predictions: min = %.4f, max = %.4f' % (np.min(predictions), np.max(predictions))

# Create a dataframe with predictions and write it to CSV file
predictions_df = pd.DataFrame(data=predictions, index=data_test.index, columns=['radiant_win'])
predictions_df.to_csv('predictions_rf.csv', sep=',')