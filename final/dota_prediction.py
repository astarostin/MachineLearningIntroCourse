import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# read training data
data = pd.read_csv('features.csv', index_col='match_id')

# optional line: pick random N rows from data for quicker testing
# data = data.ix[np.random.choice(data.index, 5000, replace=False)]

# remove summary features (except target variable)
del data['duration']
del data['tower_status_radiant']
del data['tower_status_dire']
del data['barracks_status_radiant']
del data['barracks_status_dire']
# remove useless features (like start time)
del data['start_time']

# Part 1

# Part 1.1
# get list of features that have null values in data
total_size = len(data.index)
features_with_nulls = [(data.columns.values[i], total_size - amount) for (i, amount) in enumerate(data.count())
					   if amount < total_size]
print 'Amounts of empty values by features:'
for feature in features_with_nulls:
	print '%s (%d)' % (feature[0], feature[1])

# fill null values with zeros
data = data.fillna(0)

# Part 1.2
# separate target variable and training features
Y = data['radiant_win']
del data['radiant_win']
print '\nTarget variable - radiant_win'

# Part 1.3
# Learning GradientBoostingClassifier with cross validation for 30 estimators.
# Measuring time and AUC_ROC score
kf = KFold(len(data), n_folds=5, shuffle=True)
gb = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1)
start_time = datetime.datetime.now()
scores = cross_val_score(gb, data, Y, cv=kf, scoring='roc_auc')
elapsed = datetime.datetime.now() - start_time
print '\nCross-validation for 30 trees:'
print 'time = %s, score = %.8f' % (str(elapsed), scores.mean())

# Learning GradientBoostingClassifier with cross validation and various amount of estimators.
# Analyzing AUC_ROC scores.
estimators_grid = {'n_estimators': np.arange(10, 41, 10)}
gb = GradientBoostingClassifier()
gs = GridSearchCV(gb, estimators_grid, scoring='roc_auc', cv=kf)
gs.fit(data, Y)
results = [(x.parameters['n_estimators'], x.mean_validation_score) for x in gs.grid_scores_]
for result in results:
	print '%d - %.8f' % (result[0], result[1])
print 'Best parameter n_estimators = %d, best score = %.8f' % (gs.best_params_['n_estimators'], gs.best_score_)

# Part 2

# put heroes features into separate dataframe (for future transformation)
heroes_features = ['%s%d_hero' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
heroes = data[heroes_features]

# Scaling features
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data.values), index=data.index, columns=data.columns)

# Part 2.1
# One-time logistic regression validation for time estimation
lr = LogisticRegression(C=10)
start_time = datetime.datetime.now()
scores = cross_val_score(lr, data, Y, cv=kf, scoring='roc_auc')
elapsed = datetime.datetime.now() - start_time
print '\nCross-validation for logistic regression:'
print 'time = %s, score = %.4f' % (str(elapsed), scores.mean())

# Learning LogisticRegression with cross validation and various regularization coefficient values.
# Analyzing AUC_ROC scores.
estimators_grid = {'C': [10 ** k for k in xrange(-4, 5, 1)]}
lr = LogisticRegression()
gs = GridSearchCV(lr, estimators_grid, scoring='roc_auc', cv=kf)
gs.fit(data, Y)
results = [(x.parameters['C'], x.mean_validation_score) for x in gs.grid_scores_]
for result in results:
	print '%.4f - %.8f' % (result[0], result[1])
print 'Best parameter C = %.4f, best score = %.8f' % (gs.best_params_['C'], gs.best_score_)

# Part 2.2
# remove categorical features (lobby_type, r1_hero, r2_hero, etc.) for both datasets
del data['lobby_type']
for f in heroes_features:
	del data[f]

# Learning LogisticRegression with cross validation and various regularization coefficient values.
# Analyzing AUC_ROC scores.
lr = LogisticRegression()
gs = GridSearchCV(lr, estimators_grid, scoring='roc_auc', cv=kf)
gs.fit(data, Y)
results = [(x.parameters['C'], x.mean_validation_score) for x in gs.grid_scores_]
for result in results:
	print '%.4f - %.8f' % (result[0], result[1])
print 'Best parameter C = %.4f, best score = %.8f' % (gs.best_params_['C'], gs.best_score_)

# Part 2.3
# calculate the amount of unique heroes
heroes_ids = heroes.values.ravel()
unique_ids = pd.unique(heroes_ids)
print 'Amount of unique heroes in given matches - %d' % len(unique_ids)

# Part 2.4
# Create a new list of features for heroes based on hero's id
max_hero_id = np.max(heroes_ids)
new_heroes_features_names = ['hero_%s' % str(n) for n in xrange(1, max_hero_id + 1)]
index_2_delete = []
# count feature names that were not used to remove them (max id can be greater than the number of unique ids)
for n in xrange(1, max_hero_id + 1):
	name = 'hero_%s' % str(n)
	if n not in unique_ids:
		new_heroes_features_names.remove(name)
		index_2_delete.append(n - 1)


# create a separate dataframe for new heroes features (made as a method to use for both train and test dataset)
def make_dataframe_for_new_heroes_features(old_heroes_features, max_hero_id, new_names):
	new_heroes_features = np.zeros((old_heroes_features.shape[0], max_hero_id))
	for i, match_id in enumerate(old_heroes_features.index):
		for p in xrange(5):
			new_heroes_features[i, old_heroes_features.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
			new_heroes_features[i, old_heroes_features.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
	# remove unused features (filled with zeroes)
	new_heroes_features = np.delete(new_heroes_features, index_2_delete, axis=1)
	new_df = pd.DataFrame(data=new_heroes_features, index=old_heroes_features.index, columns=new_names)
	return new_df

# make new dataframe for heroes features
new_heroes_features_df = make_dataframe_for_new_heroes_features(heroes, max_hero_id, new_heroes_features_names)
# concat 2 dataframes together along the features axis
data = pd.concat([data, new_heroes_features_df], axis=1)

# Part 2.5
# Learning LogisticRegression with cross validation and various regularization coefficient values.
# Analyzing AUC_ROC scores.
lr = LogisticRegression()
gs = GridSearchCV(lr, estimators_grid, scoring='roc_auc', cv=kf)
gs.fit(data, Y)
results = [(x.parameters['C'], x.mean_validation_score) for x in gs.grid_scores_]
for result in results:
	print '%.4f - %.8f' % (result[0], result[1])
print 'Best parameter C = %.4f, best score = %.8f' % (gs.best_params_['C'], gs.best_score_)

# Part 2.6
# Make predictions for test data
data_test = pd.read_csv('features_test.csv', index_col='match_id')
# Perform the same preparations as for training dataset
del data_test['start_time']
data_test = data_test.fillna(0)
heroes_test = data_test[heroes_features]
data_test = pd.DataFrame(scaler.fit_transform(data_test.values), index=data_test.index, columns=data_test.columns)
del data_test['lobby_type']
for f in heroes_features:
	del data_test[f]

# make new dataframe for heroes features
new_heroes_features_df_test = make_dataframe_for_new_heroes_features(heroes_test, max_hero_id, new_heroes_features_names)
# concat 2 dataframes together along the features axis
data_test = pd.concat([data_test, new_heroes_features_df_test], axis=1)

# Apply LogisticRegression with the best found value for C parameter
lr = LogisticRegression(C=gs.best_params_['C'])
# Learn on train data
lr.fit(data, Y)
# Make predictions for test data
predictions = lr.predict_proba(data_test)[:, 1]
print 'Predictions: min = %.4f, max = %.4f' % (np.min(predictions), np.max(predictions))

# Create a dataframe with predictions and write it to CSV file
predictions_df = pd.DataFrame(data=predictions, index=data_test.index, columns=['radiant_win'])
predictions_df.to_csv('predictions.csv', sep=',')
