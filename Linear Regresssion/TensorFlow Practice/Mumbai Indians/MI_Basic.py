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

x = df["Key_Runs"]
y = df["Total"]

# Plot of Training Data 
plt.scatter(x, y) 
plt.xlabel('x') 
plt.xlabel('y') 
plt.title("Training Data") 
plt.show() 



W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b") 

def Training(x,y,W,b,training_epochs,learning_rate):
	
	X = tf.placeholder("float")
	Y = tf.placeholder("float")

	n = len(x)
	# Hypothesis 
	y_pred = tf.add(tf.multiply(X, W), b)
	
	# Mean Squared Error Cost Function 
	cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n) 

	# Gradient Descent Optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

	# Global Variables Initializer 
	init = tf.global_variables_initializer() 

	# Starting the Tensorflow Session 
	with tf.Session() as sess: 
		# Initializing the Variables 
		sess.run(init) 
	
		# Iterating through all the epochs 
		for epoch in range(training_epochs): 
			
			# Feeding each data point into the optimizer using Feed Dictionary 
			for (_x, _y) in zip(x, y): 
				sess.run(optimizer, feed_dict = {X : _x, Y : _y}) 
		
			# Displaying the result after every 50 epochs 
			if (epoch + 1) % 50 == 0: 
				# Calculating the cost a every epoch 
				c = sess.run(cost, feed_dict = {X : x, Y : y}) 
				print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 
	
		# Storing necessary values to be used outside the Session 
		training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
		weight = sess.run(W) 
		bias = sess.run(b) 

	# Calculating the predictions 
	predictions = weight * x + bias 
	print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n') 

	# Plotting the Results 
	plt.plot(x, y, 'ro', label ='Original data') 
	plt.plot(x, predictions, label ='Fitted line') 
	plt.title('Linear Regression Result') 
	plt.legend() 
	plt.show()

	return weight,bias
	
Weight,Bias =  Training(x,y,W,b,learning_rate = 0.00003,training_epochs = 20)
