import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def remove_future_ahead_features(df):
	del df['duration']
	del df['tower_status_radiant']
	del df['tower_status_dire']
	del df['barracks_status_radiant']
	del df['barracks_status_dire']


def remove_useless_features(df):
	"""Remove useless features (like start time)
	:param df:
	:return:
	"""
	del df['start_time']
	del df['lobby_type']
	del df['first_blood_team']
	del df['first_blood_time']
	del df['first_blood_player1']
	del df['first_blood_player2']


def fill_na_times(df):
	times_names = ['radiant_bottle_time', 'dire_bottle_time', 'radiant_courier_time', 'dire_courier_time', \
					'radiant_flying_courier_time', 'dire_flying_courier_time', \
					'radiant_first_ward_time', 'dire_first_ward_time']
	df[times_names] = df[times_names].fillna(10000000)


def prepare_dataframe(df, test_set=False):

	# remove summary and useless features (except target variable)
	if not test_set:
		remove_future_ahead_features(df)
	remove_useless_features(df)

	# fill null values with zeros (and big values for times)
	fill_na_times(df)
	df = df.fillna(0)

	heroes_features = ['%s%d_hero' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_level = ['%s%d_level' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_xp = ['%s%d_xp' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_gold = ['%s%d_gold' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_lh = ['%s%d_lh' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_kills = ['%s%d_kills' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_items = ['%s%d_items' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_deaths = ['%s%d_deaths' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	old_times_names = [	'radiant_bottle_time', 'dire_bottle_time',\
					   	'radiant_courier_time', 'dire_courier_time',\
				   		'radiant_flying_courier_time', 'dire_flying_courier_time',\
				   		'radiant_first_ward_time', 'dire_first_ward_time']
	heroes = df[heroes_features]
	heroes_levels = df[heroes_features_level]
	heroes_xp = df[heroes_features_xp]
	heroes_gold = df[heroes_features_gold]
	heroes_lh = df[heroes_features_lh]
	heroes_kills = df[heroes_features_kills]
	heroes_items = df[heroes_features_items]
	heroes_deaths = df[heroes_features_deaths]
	old_times = df[old_times_names]

	# removing original heroes and time features
	for f in heroes_features + heroes_features_level + heroes_features_xp + heroes_features_gold + heroes_features_lh +\
		heroes_features_kills + heroes_features_items + heroes_features_deaths + old_times_names:
		del df[f]

	# Scaling features
	if not test_set:
		df.to_csv('before_scaling.csv')

	scaler = StandardScaler()
	df = pd.DataFrame(scaler.fit_transform(df.values), index=df.index, columns=df.columns)

	if not test_set:
		df.to_csv('after_scaling.csv')

	# new additive feature names
	heroes_additive_feature_names = ['%s_%s' % (team, f) for team in ['r', 'd'] for f in ['level', 'xp', 'gold', 'lh',
																						  'kills', 'items', 'deaths']]
	# new times feature names
	times_feature_names = ['bottle_time', 'courier_time', 'flying_courier_time',  'first_ward_time']

	# calculate the amount of unique heroes
	heroes_ids = heroes.values.ravel()
	unique_ids = pd.unique(heroes_ids)

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

	new_heroes_features = np.zeros((df.shape[0], max_hero_id))
	heroes_additive_features = np.zeros((df.shape[0], len(heroes_additive_feature_names)))
	times_features = np.zeros((df.shape[0], len(times_feature_names)))
	for i, match_id in enumerate(df.index):
		# process heroes id features
		for p in xrange(5):
			new_heroes_features[i, heroes.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
			new_heroes_features[i, heroes.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

		# process heroes additive features
		heroes_additive_features[i, 0] = heroes_levels.ix[match_id, :5].sum()
		heroes_additive_features[i, 1] = heroes_xp.ix[match_id, :5].sum()
		heroes_additive_features[i, 2] = heroes_gold.ix[match_id, :5].sum()
		heroes_additive_features[i, 3] = heroes_lh.ix[match_id, :5].sum()
		heroes_additive_features[i, 4] = heroes_kills.ix[match_id, :5].sum()
		heroes_additive_features[i, 5] = heroes_items.ix[match_id, :5].sum()
		heroes_additive_features[i, 6] = heroes_deaths.ix[match_id, :5].sum()
		heroes_additive_features[i, 7] = heroes_levels.ix[match_id, 5:].sum()
		heroes_additive_features[i, 8] = heroes_xp.ix[match_id, 5:].sum()
		heroes_additive_features[i, 9] = heroes_gold.ix[match_id, 5:].sum()
		heroes_additive_features[i, 10] = heroes_lh.ix[match_id, 5:].sum()
		heroes_additive_features[i, 11] = heroes_kills.ix[match_id, 5:].sum()
		heroes_additive_features[i, 12] = heroes_items.ix[match_id, 5:].sum()
		heroes_additive_features[i, 13] = heroes_deaths.ix[match_id, 5:].sum()

		#proces team times features
		for j in xrange(0, 8, 2):
			time_diff = old_times.ix[match_id, j] - old_times.ix[match_id, j + 1]
			if time_diff < 0:
				times_features[i, j/2] = 1
			elif time_diff > 0:
				times_features[i, j/2] = -1

	# remove unused features
	new_heroes_features = np.delete(new_heroes_features, index_2_delete, axis=1)

	# create new dataframes
	new_heroes_features_df = pd.DataFrame(data=new_heroes_features, index=df.index, columns=new_heroes_features_names)
	new_heroes_additive_features_df = pd.DataFrame(data=scaler.fit_transform(heroes_additive_features), index=df.index, columns=heroes_additive_feature_names)
	new_times_features_df = pd.DataFrame(data=times_features, index=df.index, columns=times_feature_names)

	# concat all dataframes together along the features axis
	df = pd.concat([df, new_heroes_features_df], axis=1)
	df = pd.concat([df, new_heroes_additive_features_df], axis=1)
	df = pd.concat([df, new_times_features_df], axis=1)
	return df


# read training data
data = pd.read_csv('features.csv', index_col='match_id')

# optional line: pick random N rows from data for quicker testing
# data = data.ix[np.random.choice(data.index, 3000, replace=False)]

# separate target variable and training features
Y = data['radiant_win']
del data['radiant_win']

data = prepare_dataframe(data)
data.to_csv('train_prepared.csv', sep=',')

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
data_test = pd.read_csv('features_test.csv', index_col='match_id')
# Perform the same preparations as for training dataset
data_test = prepare_dataframe(data_test, test_set=True)
data_test.to_csv('test_prepared.csv', sep=',')

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