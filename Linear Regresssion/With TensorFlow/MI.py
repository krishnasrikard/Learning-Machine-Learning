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

df = pd.read_csv('../Dataset/MI.csv',sep=',')															#Importing the .csv for data analysis
df.reindex(np.random.permutation(df.index))													#Shuffle																	
print (df)
print ("--------------------------------------------------------------------------")
print (df.describe())																		#Gives statitics of the data
print ("--------------------------------------------------------------------------")
df['Opening'] = df['Rohit'] + df['Quinton']													#Creating a new column
print (df)
print ("--------------------------------------------------------------------------")

Input = df[["Opening"]]																		#Considering Input
Input_columns = [tf.feature_column.numeric_column("Opening")]								#Feeding Input columns
Targets = df["Total"]																		#Considering Output

Current_Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)					#Setting Parameters for Optimiser			
Current_Optimizer = tf.contrib.estimator.clip_gradients_by_norm(Current_Optimizer, 5.0)

Linear_Regressor = tf.estimator.LinearRegressor(feature_columns = Input_columns,optimizer = Current_Optimizer)		#Feeding Linear Regressor


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):			#Defining Input Function as Dictionaries
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
    
    
Training = Linear_Regressor.train(input_fn = lambda:my_input_fn(Input, Targets), steps=100)		#Training 

# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't 
# need to repeat or shuffle the data here.
Prediction_Input_fn = lambda: my_input_fn(Input, Targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
Predictions = Linear_Regressor.predict(input_fn=Prediction_Input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
Predictions = np.array([item['predictions'][0] for item in Predictions])

# Print Mean Squared Error and Root Mean Squared Error.
Mean_Squared_Error = metrics.mean_squared_error(Predictions, Targets)
Root_Mean_Squared_Error = math.sqrt(Mean_Squared_Error)
print("Mean Squared Error (on training data): %0.3f" % Mean_Squared_Error)
print("Root Mean Squared Error (on training data): %0.3f" % Root_Mean_Squared_Error)


Min_Total_Value = df["Total"].min()
Max_Total_Value = df["Total"].max()
Min_Max_difference = Max_Total_Value - Min_Total_Value

print("Min. Total: %0.3f" % Min_Total_Value)
print("Max. Total: %0.3f" % Max_Total_Value)
print("Difference between Min. and Max.: %0.3f" % Min_Max_difference)
print("Root Mean Squared Error: %0.3f" % Root_Mean_Squared_Error)


Calibration_Data = pd.DataFrame()
Calibration_Data["Predictions"] = pd.Series(Predictions)
Calibration_Data["Targets"] = pd.Series(Targets)
print (Calibration_Data.describe())

Sample = df.sample(n = 16)


# Get the min and max total_rooms values.
x_0 = Sample["Opening"].min()
x_1 = Sample["Opening"].max()

# Retrieve the final weight and bias generated during training.
weight = Linear_Regressor.get_variable_value('linear/linear_model/Opening/weights')[0]
bias = Linear_Regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c ='r')

# Label the graph axes.
plt.ylabel("Total")
plt.xlabel("Opening")

# Plot a scatter plot from our data sample.
plt.scatter(Sample["Opening"], Sample["Total"])

# Display graph.
plt.show()
