"""
Implementation of Support Vector Machine using Tensorflow 1.x
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs
from sklearn import cluster, datasets
import seaborn as sns
import imageio
import time
from IPython.display import HTML
import warnings
from numpy import linalg as LA
import tensorflow as tf

print ("--------------------------------------------Imported Libraries---------------------------------")

# Setting cmap colors
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","green","blue"])
np.random.seed(0)

# Generating Dataset
data, labels = make_blobs(n_samples=750, n_features=2, centers=3, cluster_std=0.5, random_state=0)
print(data.shape, labels.shape)

# Visualising Data
plt.scatter(data[:,0], data[:,1], c=labels, cmap=my_cmap)
plt.show()

# Defining Positive and Negetive Classes

Class1_y = np.array([1 if y==0 else -1 for y in labels])
Class2_y = np.array([1 if y==1 else -1 for y in labels])
Class3_y = np.array([1 if y==2 else -1 for y in labels])
y = np.array([Class1_y, Class2_y, Class3_y])

Class1_X = [x[0] for i,x in enumerate(data) if labels[i]==0]
Class1_Y = [x[1] for i,x in enumerate(data) if labels[i]==0]
Class2_X = [x[0] for i,x in enumerate(data) if labels[i]==1]
Class2_Y = [x[1] for i,x in enumerate(data) if labels[i]==1]
Class3_X = [x[0] for i,x in enumerate(data) if labels[i]==2]
Class3_Y = [x[1] for i,x in enumerate(data) if labels[i]==2]


# Gaussian Similarity Function
def Similarity_Gaussian(x,l,Variance = 2):
	f = np.exp(-1*pow((LA.norm(x-l)),2)/(2*Variance))
	return f


# Class for SVM	
class SVM():
	
	def __init__(self, kernal='Gaussian_Kernal', no_features=2, no_classes=3,C = 0.1):
		self.kernal = kernal
		self.C = C
		self.W = None
		self.no_classes = no_classes
		self.Kernal = None
		self.no_features = no_features
		self.Variance = None
	
	
	# Part of Cost Function
	def Cost0(self,W,F):
		x = tf.matmul(W,F)
		x = tf.numpy_function(np.where, [x>-1,x+1,0], tf.float32)
		return x
		
		
	# Part of Cost Function
	def Cost1(self,W,F):
		x = tf.matmul(W,F)
		x = tf.numpy_function(np.where, [x<1,-x+1,0], tf.float32)
		return x		
		
		
	# Function to Compute Cost Function
	def ComputeCost(self,X,y_true):
		c = self.no_classes
		
		self.Kernal = self.ComputeKernal(X)
		
		Reg_Term = tf.multiply(0.5, tf.reduce_sum(tf.multiply(self.W,self.W)))
		Loss_term1 = tf.multiply(y_true,self.Cost1(self.W,self.Kernal))
		Loss_term2 = tf.multiply((1-y_true),self.Cost0(self.W,self.Kernal))
		Loss_Term = self.C * tf.reduce_sum(tf.add(Loss_term1,Loss_term2))
		
		Loss = tf.add(Loss_Term,Reg_Term)
				
		return Loss
		
		
	# Function to Compute Accuracy
	def ComputeAccuracy(self,X,X_test,y,Variance):
		
		self.Kernal = self.ComputeKernal(X)
		
		a = tf.reshape(tf.reduce_sum(tf.square(X), 1),[-1,1])
		b = tf.reshape(tf.reduce_sum(tf.square(X_test), 1),[-1,1])
		Distance_Predicted = tf.add(tf.subtract(a, tf.multiply(2., tf.matmul(X,tf.transpose(X_test)))), tf.transpose(b))
		Predicted_Kernal = tf.exp(- 1 * tf.abs(Distance_Predicted) / (2*Variance))
				
		Prediction = tf.matmul(tf.multiply(y,self.W), Predicted_Kernal)
		Prediction = tf.arg_max(Prediction - tf.expand_dims(tf.reduce_mean(Prediction,1), 1), 0)
		Accuracy = tf.reduce_mean(tf.cast(tf.equal(Prediction,tf.argmax(y,0)), tf.float32))
		
		return tf.math.round(Accuracy*100)
		
	
	# Generating Kernals
	def ComputeKernal(self,X):
		# Creating Gaussian Kernal
		if self.kernal == 'Gaussian_Kernal':
			d = tf.reduce_sum(tf.square(X), axis=1)
			d = tf.reshape(d, [-1,1])
			d_norm = tf.add(tf.subtract(d, tf.multiply(2., tf.matmul(X,tf.transpose(X)))), tf.transpose(d))
			K = tf.exp(-1*tf.abs(d_norm)/2*self.Variance)
			
		# Creating Linear Kernal
		elif self.kernal == 'Linear_Kernal':
			K = tf.matmul(X,tf.transpose(X))
		
		return K
		
	
	# Fit the Training Data
	def SVMFit (self,X_train,Y_train,Epochs,Variance,Plot_Loss=True):
		
		sess = tf.Session()
		
		self.Variance = Variance		
		c = self.no_classes
		f = self.no_features
		self.W = tf.Variable(tf.random.normal(shape=[c,750]))
		X = tf.placeholder(shape=[None, f], dtype=tf.float32)
		y = tf.placeholder(shape=[c, None], dtype=tf.float32)
		X_test = tf.placeholder(shape=[None, f], dtype=tf.float32)
		
		# No.of Training Examples
		m = X.shape[0]
		# No.of Features
		n = X.shape[1]
		
		# LandMarks
		L = X
		
		Loss_Data = []
		Accuracy_Data = []
		
		loss = self.ComputeCost(X,y)
		accuracy = self.ComputeAccuracy(X,X_test,y,self.Variance)
		
		# Declaring Optimizer
		Optimizer = tf.train.AdamOptimizer(0.001)
		Train_Step = Optimizer.minimize(loss)
		init = tf.initialize_all_variables()
		sess.run(init)
		
		for i in range(Epochs):
			print ("Epoch - {} :".format(i+1))
			Xi = np.reshape(X_train,[-1,2])
			Yi = np.reshape(Y_train,[3,-1])
			sess.run(Train_Step, feed_dict={X:Xi, y:Yi})
			Epoch_Loss = sess.run(loss, feed_dict={X:Xi, y:Yi})
			Loss_Data.append(Epoch_Loss)
			Epoch_Acc = sess.run(accuracy, feed_dict={X:Xi, y:Yi, X_test:Xi})
			Accuracy_Data.append(Epoch_Acc)
			
			print ("Loss = {} and Accuracy = {}".format(Epoch_Loss,Epoch_Acc))
			
		A = Epoch_Acc
		
		# Plotting Loss
		if Plot_Loss==True:
			plt.plot(Loss_Data)
			plt.title("Loss Data during Training")
			plt.show()
		
		return A
		
	def Predict(self,X,x,y):
		
		X_test = np.zeros((x.shape[0],2))
		X_test[:,0] = x
		X_test[:,1] = y
		
		X = tf.Variable(X,dtype=tf.float32)
		X_test = tf.Variable(X_test,dtype=tf.float32)
		
		a = tf.reshape(tf.reduce_sum(tf.square(X), 1),[-1,1])
		b = tf.reshape(tf.reduce_sum(tf.square(X_test), 1),[-1,1])
		Distance_Predicted = tf.add(tf.subtract(a, tf.multiply(2., tf.matmul(X,tf.transpose(X_test)))), tf.transpose(b))
		Predicted_Kernal = tf.exp(- 1 * tf.abs(Distance_Predicted) / (2*self.Variance))
		
		Prediction = tf.matmul(self.W, Predicted_Kernal)
		
		return Prediction

model = SVM()
Accuracy = model.SVMFit(data,y,500,1)
print ("Final Accuracy is ", Accuracy)

"""
# Verifying Data
h = 0.02
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.Predict(data,xx.ravel(),yy.ravel())
Z = tf.math.argmax(Z,axis=0)
Z = tf.reshape(Z,(xx.shape[0],xx.shape[1]))

plt.contour([xx, yy], Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.show()
"""
