"""
The following code utilises TesnsorFlow 1.x
"""
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





def train_model(learning_rate, steps, batch_size, input_feature = "Key_Runs"):			#Defining a function to train model
  """Trains a linear regression model of one feature.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `california_housing_dataframe`
      to use as input feature.
  """
  
  periods = 10																			#Setting no.of periods
  steps_per_period = steps / periods

  Input_label = input_feature															#Setting Input Feature
  Input_data = df[[Input_label]]						
  Output_label = "Total"																#Setting Output Label
  Targets = df[Output_label]

  # Create feature columns.
  Input_columns = [tf.feature_column.numeric_column(Input_label)]						#Feeding Input to the Model
  
  # Create input functions.
  Training_Input_fn = lambda:my_input_fn(Input_data, Targets, batch_size=batch_size)
  Prediction_Input_fn = lambda:my_input_fn(Input_data, Targets, num_epochs=1, shuffle=False)
  
  # Create a linear regressor object.
  Current_Optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  Current_Optimizer = tf.contrib.estimator.clip_gradients_by_norm(Current_Optimizer, 5.0)
  Linear_Regressor = tf.estimator.LinearRegressor(feature_columns=Input_columns,optimizer=Current_Optimizer)

  # Set up to plot the state of our model's line each period.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(Output_label)
  plt.xlabel(Input_label)
  Sample = df.sample(n=10)
  plt.scatter(Sample[Input_label], Sample[Output_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    Linear_Regressor.train(input_fn=Training_Input_fn,steps=steps_per_period)
    # Take a break and compute predictions.
    Predictions = Linear_Regressor.predict(input_fn=Prediction_Input_fn)
    Predictions = np.array([item['predictions'][0] for item in Predictions])
    
    # Compute loss.
    root_mean_squared_error = math.sqrt(metrics.mean_squared_error(Predictions, Targets))
    # Occasionally print the current loss.
    print("period %02d : %0.2f" % (period, root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    
    
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, Sample[Output_label].max()])
    
    weight = Linear_Regressor.get_variable_value('linear/linear_model/%s/weights' % Input_label)[0]
    bias = Linear_Regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,Sample[Input_label].max()),Sample[Input_label].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period])
  print("Model Training Finished.")

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)

  # Output a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["Predictions"] = pd.Series(Predictions)
  calibration_data["Targets"] = pd.Series(Targets)
  display.display(calibration_data.describe())

  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
  return calibration_data
  
  
df["Strike"] = df["Key_Runs"] / df["Opening"].apply(lambda x: min(x,5))
print (df)
print ("--------------------------------------------------------------------------")

calibration_data = train_model(learning_rate=0.15,steps=1000,batch_size=11,input_feature="Strike")

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(calibration_data["Predictions"], calibration_data["Targets"])

plt.subplot(1, 2, 2)
k = df["Strike"].hist()																			#Creates Histogram
print (k)

plt.show()
