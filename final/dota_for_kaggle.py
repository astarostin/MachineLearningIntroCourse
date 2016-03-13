import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def remove_future_ahead_features(df):
	future_ahead_features = ['duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', \
							 'barracks_status_dire']

	df.drop(future_ahead_features, axis=1, inplace=True)


def remove_useless_features(df):
	"""Remove useless features (like start time)
	:param df:
	:return:
	"""
	# maybe also 'first_blood_team'
	useless_features = ['start_time', 'lobby_type', 'first_blood_time', 'first_blood_player1', 'first_blood_player2']
	df.drop(useless_features, axis=1, inplace=True)


def fill_na_values(df):
	times_names = ['radiant_bottle_time', 'dire_bottle_time', 'radiant_courier_time', 'dire_courier_time', \
				   'radiant_flying_courier_time', 'dire_flying_courier_time', \
				   'radiant_first_ward_time', 'dire_first_ward_time']
	df[times_names] = df[times_names].fillna(10000000)
	df['first_blood_team'].replace(1, -1, inplace=True)
	df['first_blood_team'].replace(0, 1, inplace=True)
	df['first_blood_team'].fillna(0, inplace=True)
	df.fillna(0, inplace=True)


def prepare_dataframe(df, test_set=False):
	# remove summary and useless features (except target variable)
	if not test_set:
		remove_future_ahead_features(df)
	remove_useless_features(df)

	# fill null values with zeros (and big values for times)
	fill_na_values(df)

	heroes_features_id = ['%s%d_hero' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_level = ['%s%d_level' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_xp = ['%s%d_xp' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_gold = ['%s%d_gold' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_lh = ['%s%d_lh' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_kills = ['%s%d_kills' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_items = ['%s%d_items' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	heroes_features_deaths = ['%s%d_deaths' % (team, n) for team in ['r', 'd'] for n in xrange(1, 6)]
	items_features_times = ['radiant_bottle_time', 'dire_bottle_time', \
					   'radiant_courier_time', 'dire_courier_time', \
					   'radiant_flying_courier_time', 'dire_flying_courier_time', \
					   'radiant_first_ward_time', 'dire_first_ward_time']
	# items_features_counts = ['radiant_tpscroll_count', 'radiant_boots_count', 'radiant_ward_observer_count', \
	# 						'radiant_ward_sentry_count', 'dire_tpscroll_count', 'dire_boots_count', \
	# 						'dire_ward_observer_count', 'dire_ward_sentry_count']


	heroes_id = df[heroes_features_id]
	heroes_levels = df[heroes_features_level]
	heroes_xp = df[heroes_features_xp]
	heroes_gold = df[heroes_features_gold]
	heroes_lh = df[heroes_features_lh]
	heroes_kills = df[heroes_features_kills]
	heroes_items = df[heroes_features_items]
	heroes_deaths = df[heroes_features_deaths]
	items_times = df[items_features_times]
	first_blood_team = df['first_blood_team']
	# items_counts = df[items_features_counts]

	# removing original heroes and time features
	df.drop(heroes_features_id + heroes_features_level + heroes_features_xp + heroes_features_gold + \
			heroes_features_lh + heroes_features_kills + heroes_features_items + heroes_features_deaths + \
			# items_features_counts + \
			items_features_times + ['first_blood_team'], axis=1, inplace=True)

	# Scaling features
	if not test_set:
		df.to_csv('before_scaling.csv')

	scaler = MinMaxScaler((0, 10))
	#scaler = StandardScaler()
	df = pd.DataFrame(scaler.fit_transform(df.values), index=df.index, columns=df.columns)

	if not test_set:
		df.to_csv('after_scaling.csv')

	# new additive feature names
	additive_feature_names = ['%s_%s' % (team, f) for team in ['r', 'd'] for f in ['level', 'xp', 'gold', 'lh',
																						  'kills', 'items', 'deaths']]
	#additive_total_feature_names = ['%s_level_xp_gold_items' % team for team in ['r', 'd']]
	# counts_total_feature_names = ['%s_counts' % team for team in ['r', 'd']]
	#killed_total_feature_names = ['%s_lh_kills' % team for team in ['r', 'd']]
	#deaths_total_feature_names = ['%s_deaths' % team for team in ['r', 'd']]

	# new times feature names
	times_feature_names = ['bottle_time', 'courier_time', 'flying_courier_time', 'first_ward_time']

	# calculate the amount of unique heroes
	heroes_ids = heroes_id.values.ravel()
	unique_ids = pd.unique(heroes_ids)

	# Create a new list of features for heroes based on hero's id
	max_hero_id = np.max(heroes_ids)
	heroes_features_names = ['hero_%s' % str(n) for n in xrange(1, max_hero_id + 1)]
	index_2_delete = []
	# count feature names that were not used to remove them (max id can be greater than the number of unique ids)
	for n in xrange(1, max_hero_id + 1):
		name = 'hero_%s' % str(n)
		if n not in unique_ids:
			heroes_features_names.remove(name)
			index_2_delete.append(n - 1)

	heroes_features = np.zeros((df.shape[0], max_hero_id))
	additive_features = np.zeros((df.shape[0], len(additive_feature_names)))
	#additive_total_features = np.zeros((df.shape[0], len(additive_total_feature_names)))
	# counts_total_features = np.zeros((df.shape[0], len(counts_total_feature_names)))
	#killed_total_features = np.zeros((df.shape[0], len(killed_total_feature_names)))
	#deaths_total_features = np.zeros((df.shape[0], len(deaths_total_feature_names)))
	times_features = np.zeros((df.shape[0], len(times_feature_names)))

	total = len(df.index)
	for i, match_id in enumerate(df.index):
		# process heroes id features
		for p in xrange(5):
			heroes_features[i, heroes_id.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
			heroes_features[i, heroes_id.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

		# process heroes additive features
		additive_features[i, 0] = heroes_levels.ix[match_id, :5].sum()
		additive_features[i, 1] = heroes_xp.ix[match_id, :5].sum()
		additive_features[i, 2] = heroes_gold.ix[match_id, :5].sum()
		additive_features[i, 3] = heroes_lh.ix[match_id, :5].sum()
		additive_features[i, 4] = heroes_kills.ix[match_id, :5].sum()
		additive_features[i, 5] = heroes_items.ix[match_id, :5].sum()
		additive_features[i, 6] = heroes_deaths.ix[match_id, :5].sum()
		additive_features[i, 7] = heroes_levels.ix[match_id, 5:].sum()
		additive_features[i, 8] = heroes_xp.ix[match_id, 5:].sum()
		additive_features[i, 9] = heroes_gold.ix[match_id, 5:].sum()
		additive_features[i, 10] = heroes_lh.ix[match_id, 5:].sum()
		additive_features[i, 11] = heroes_kills.ix[match_id, 5:].sum()
		additive_features[i, 12] = heroes_items.ix[match_id, 5:].sum()
		additive_features[i, 13] = heroes_deaths.ix[match_id, 5:].sum()

		# process additive total features
		# additive_total_features[i, 0] = heroes_levels.ix[match_id, :5].sum() + heroes_xp.ix[match_id, :5].sum() + \
		# 								heroes_gold.ix[match_id, :5].sum() + heroes_items.ix[match_id, :5].sum()
		# additive_total_features[i, 1] = heroes_levels.ix[match_id, 5:].sum() + heroes_xp.ix[match_id, 5:].sum() + \
		# 								heroes_gold.ix[match_id, 5:].sum() + heroes_items.ix[match_id, 5:].sum()

		# process counts total features items_counts
		# counts_total_features[i, 0] = items_counts.ix[match_id, :4].sum()
		# counts_total_features[i, 1] = items_counts.ix[match_id, 4:].sum()

		# process killed total features
		# killed_total_features[i, 0] = heroes_lh.ix[match_id, :5].sum() + heroes_kills.ix[match_id, :5].sum()
		# killed_total_features[i, 1] = heroes_lh.ix[match_id, 5:].sum() + heroes_kills.ix[match_id, 5:].sum()

		# process deaths total features
		# deaths_total_features[i, 0] = heroes_deaths.ix[match_id, :5].sum()
		# deaths_total_features[i, 1] = heroes_deaths.ix[match_id, 5:].sum()

		# proces team times features
		for j in xrange(0, 8, 2):
			time_diff = items_times.ix[match_id, j] - items_times.ix[match_id, j + 1]
			# what time_diff to count as considerable
			if np.abs(time_diff) >= 60:
				if time_diff < 0:
					times_features[i, j / 2] = 1
				elif time_diff > 0:
					times_features[i, j / 2] = -1
		if i % 1000 == 0:
			print '%d%% done' % ((float(i) / total) * 100)

	# remove unused features
	heroes_features = np.delete(heroes_features, index_2_delete, axis=1)

	# create new dataframes
	heroes_features_df = pd.DataFrame(data=heroes_features, index=df.index, columns=heroes_features_names)
	additive_features_df = pd.DataFrame(data=scaler.fit_transform(additive_features), index=df.index,
												   columns=additive_feature_names)
	# additive_total_features_df = pd.DataFrame(data=scaler.fit_transform(additive_total_features), index=df.index,
	# 											   columns=additive_total_feature_names)
	# counts_total_features_df = pd.DataFrame(data=scaler.fit_transform(counts_total_features), index=df.index,
	# 											   columns=counts_total_feature_names)
	# killed_total_features_df = pd.DataFrame(data=scaler.fit_transform(killed_total_features), index=df.index,
	# 											   columns=killed_total_feature_names)
	# deaths_total_features_df = pd.DataFrame(data=scaler.fit_transform(deaths_total_features), index=df.index,
	# 											   columns=deaths_total_feature_names)
	times_features_df = pd.DataFrame(data=times_features, index=df.index, columns=times_feature_names)

	# concat all dataframes together along the features axis
	df = pd.concat([df, heroes_features_df, additive_features_df, times_features_df], axis=1)
	# df = pd.concat([df, heroes_features_df, additive_total_features_df, counts_total_features_df, \
	# 				killed_total_features_df, deaths_total_features_df, times_features_df], axis=1)
	df['first_blood_team'] = first_blood_team
	return df


# read training data
data = pd.read_csv('features.csv', index_col='match_id')

# optional line: pick random N rows from data for quicker testing
#data = data.ix[np.random.choice(data.index, 100, replace=False)]
#data = data.ix[:9999]

# separate target variable and training features
Y = data['radiant_win']
data.drop('radiant_win', axis=1, inplace=True)

data = prepare_dataframe(data)

data['radiant_win'] = Y
data.to_csv('train_prepared.csv', sep=',')
data.drop('radiant_win', axis=1, inplace=True)

# Learning LogisticRegression with cross validation and various regularization coefficient values.
# Analyzing AUC_ROC scores.
start_time = datetime.datetime.now()
kf = KFold(len(data), n_folds=5, shuffle=True, random_state=241)
c_param = 0.0650
lr = LogisticRegression(C=c_param, random_state=241)
scores = cross_val_score(lr, data, Y, cv=kf, scoring='roc_auc')
elapsed = datetime.datetime.now() - start_time
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
