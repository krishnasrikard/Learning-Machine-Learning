from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


df = pd.read_csv('MI.csv',sep=',')															#Importing the .csv for data analysis
df = df.reindex(np.random.permutation(df.index))											#Shuffle																	
print (df)
print ("--------------------------------------------------------------------------")
print (df.describe())																		#Gives statitics of the data
print ("--------------------------------------------------------------------------")
df['Opening'] = df['Rohit'] + df['Quinton']
df['Key_Runs'] = df['Rohit'] + df['Quinton'] + df['Surya'] + df['Pollard'] + df['Krunal'] + df['Hardik']		#Creating a column
print (df)
print ("--------------------------------------------------------------------------")


def preprocess_features(df):
	"""Prepares input features from MI set.
	Args:
	Selected_Players: A Pandas DataFrame expected to contain data from the MI data set.
	Returns:
	A DataFrame that contains the features to be used for the model, including synthetic features.
	"""
	Selected_Players = df[["Rohit","Quinton","Surya","Krunal","Pollard","Hardik"]]
	processed_features = Selected_Players.copy()
	# Create a synthetic feature.
	processed_features["Strike"] = (df["Key_Runs"] / df["Opening"])
	
	return processed_features


def preprocess_targets(df):
	"""Prepares target features (i.e., labels) from MI set.
	Args:
    Selected_Players: A Pandas DataFrame expected to contain data from the MI data set.
	Returns:
	A DataFrame that contains the target feature.
	"""
	
	output_targets = pd.DataFrame()
	output_targets["Result"] = df["Result"]
	
	return output_targets



training_examples = preprocess_features(df.head(23))										#First 23 of the data will be for Training
print (training_examples.describe())
print ("--------------------------------------------------------------------------")

training_targets = preprocess_targets(df.head(23))
print (training_targets.describe())
print ("--------------------------------------------------------------------------")

validation_examples = preprocess_features(df.tail(7))										#Last 7 of the data will be for Validation
print (validation_examples.describe())
print ("--------------------------------------------------------------------------")

validation_targets = preprocess_targets(df.tail(7))
print (validation_targets.describe())
print ("--------------------------------------------------------------------------")


def my_input_fn(features, targets, batch_size = 1, shuffle=True, num_epochs=None):			#Defining a input function
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def get_quantile_based_buckets(feature_values, num_buckets):
	quantiles = feature_values.quantile([(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
	
	return [quantiles[q] for q in quantiles.keys()]
	

def construct_feature_columns():
	"""Construct the TensorFlow Feature Columns.
	Returns:
		A set of feature columns
	""" 
	Rohit = tf.feature_column.numeric_column("Rohit")
	Quinton = tf.feature_column.numeric_column("Quinton")
	Surya = tf.feature_column.numeric_column("Surya")
##	Ishan = tf.feature_column.numeric_column("Ishan")
	Pollard = tf.feature_column.numeric_column("Pollard")
	Hardik = tf.feature_column.numeric_column("Hardik")
	Krunal = tf.feature_column.numeric_column("Krunal")

	bucketized_Rohit = tf.feature_column.bucketized_column(Rohit, boundaries=get_quantile_based_buckets(df["Rohit"], 4))
	bucketized_Quinton = tf.feature_column.bucketized_column(Quinton, boundaries=get_quantile_based_buckets(df["Quinton"], 4))			# Divide Quinton into 4 buckets.
	bucketized_Surya = tf.feature_column.bucketized_column(Surya, boundaries=get_quantile_based_buckets(df["Surya"], 5))					# Divide Surya into 5 buckets.
##	bucketized_Ishan = tf.feature_column.bucketized_column(Ishan, boundaries=get_quantile_based_buckets(df["Ishan"], 3))
	bucketized_Pollard = tf.feature_column.bucketized_column(Pollard, boundaries=get_quantile_based_buckets(df["Pollard"], 4))
	bucketized_Krunal = tf.feature_column.bucketized_column(Krunal, boundaries=get_quantile_based_buckets(df["Krunal"], 3))
	bucketized_Hardik = tf.feature_column.bucketized_column(Hardik, boundaries=get_quantile_based_buckets(df["Hardik"], 3))
	bucketized_Opening = tf.feature_column.crossed_column(set([bucketized_Rohit, bucketized_Quinton]), hash_bucket_size = 1000)

	
	feature_columns = set([bucketized_Rohit,bucketized_Quinton,bucketized_Surya,bucketized_Pollard,bucketized_Krunal,bucketized_Hardik,bucketized_Opening])
  
	return feature_columns

def model_size(estimator):
	variables = estimator.get_variable_names()
	size = 0
	for variable in variables:
		if not any(x in variable 
				for x in ['global_step','centered_bias_weight','bias_weight','Ftrl']):
			size += np.count_nonzero(estimator.get_variable_value(variable))
	
	return size


def train_linear_classifier_model(learning_rate,regularization_strength,steps,batch_size,feature_columns,training_examples,training_targets,validation_examples,validation_targets):
	"""Trains a linear regression model of multiple features.
  
	In addition to training, this function also prints training progress information,
	as well as a plot of the training and validation loss over time.
  
	Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `MI` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `MI` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `MI` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `MI` to use as target for validation.
      
	Returns:
		A `LinearRegressor` object trained on the training data.
	"""

	periods = 10
	steps_per_period = steps / periods
  
	# Create a linear regressor object.
	Current_Optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
	Current_Optimizer = tf.contrib.estimator.clip_gradients_by_norm(Current_Optimizer, 5.0)
	Linear_Classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, optimizer=Current_Optimizer)
  
	# 1. Create input functions.
	training_input_fn = lambda: my_input_fn(training_examples,training_targets["Result"],batch_size=batch_size)
	predict_training_input_fn = lambda: my_input_fn(training_examples,training_targets["Result"],num_epochs=1,shuffle=False)
	predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets["Result"],num_epochs=1,shuffle=False)

	# Train the model, but do so inside a loop so that we can periodically assess loss metrics.
	print("Training model...")
	print("LogLoss (on training data):")
	training_log_losses = []
	validation_log_losses = []
	for period in range (0, periods):
		# Train the model, starting from the prior state.
		Linear_Classifier.train(input_fn=training_input_fn,steps=steps_per_period)
		# Take a break and compute predictions.    
		training_probabilities = Linear_Classifier.predict(input_fn=predict_training_input_fn)
		training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
	
		validation_probabilities = Linear_Classifier.predict(input_fn=predict_validation_input_fn)
		validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
	
		training_log_loss = metrics.log_loss(training_targets, training_probabilities)
		validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
		# Occasionally print the current loss.
		print("  period %02d : %0.2f" % (period, training_log_loss))
		# Add the loss metrics from this period to our list.
		training_log_losses.append(training_log_loss)
		validation_log_losses.append(validation_log_loss)
	
	print("Model training finished.")
  
	# Output a graph of loss metrics over periods.
	plt.ylabel("LogLoss")
	plt.xlabel("Periods")
	plt.title("LogLoss vs. Periods")
	plt.tight_layout()
	plt.plot(training_log_losses, label="training")
	plt.plot(validation_log_losses, label="validation")
	plt.legend()

	return Linear_Classifier


linear_classifier = train_linear_classifier_model(
	learning_rate=0.1,
	regularization_strength=0.1,
	steps=300,
	batch_size=100,
	feature_columns=construct_feature_columns(),
	training_examples=training_examples,
	training_targets=training_targets,
	validation_examples=validation_examples,
	validation_targets=validation_targets)

print("Model size:", model_size(linear_classifier))
