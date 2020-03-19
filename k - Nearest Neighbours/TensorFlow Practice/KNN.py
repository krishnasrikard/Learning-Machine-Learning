"""
Implementation of k-Nearest Neighbours Algorithm
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

print ("--------------------------------------------Imported Libraries---------------------------------")

# Add Columns to Pandas DataFrame
def Add_Columns(df):
	# Edit this for each Dataset depending on column of Output
	n = len(df.columns) - 1
	Columns = []
	for i in range(n):
		Columns.append("Data-"+str(i+1))
	Columns.append("Labels")
	
	df.columns = Columns
	
	return df
	
# Importing Data
df = pd.read_csv('Iris_Flower_Data.csv', delimiter=',')	
df = Add_Columns(df)									
df = df.reindex(np.random.permutation(df.index))						#Shuffle																	
print (df)
print ("--------------------------------------------------------------------------")
print (df.describe())													#Gives statitics of the data
print ("--------------------------------------------------------------------------")


# Handling Labels
Labels = np.unique(df["Labels"])										# Extracting Unique Data
Label_Map = {}
for i,label in enumerate(Labels):
	Label_Map[label] = i												# Assigning Index to each Label
	
print ("Label Mapping is ",Label_Map)
df['Labels'] = [Label_Map[l] for l in df['Labels']]						# Update Labels
print ("--------------------------------------------------------------------------")

# Splitting Data
X = np.array(df.loc[:, df.columns != 'Labels'])							# Training Inputs
Y = np.array(df["Labels"])												# Training Outputs/Targets
X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.33,stratify=Y)
print ("Size of Training Set ",X_train.shape)
print ("Size of Validation Set ",X_val.shape)
print ("--------------------------------------------------------------------------")

# Class for KNN
class KNN():
	def __init__(self,k):
		self.k = k
		
	def EuclidenDistance(self,X,y):
		# Calculate Eucliden Distance
		d = np.linalg.norm(X-y,axis=1)
		return d
		
	def NormaliseData(self,Data):
		# Normalise Data
		Data[:,0:4] = sklearn.preprocessing.normalize(Data[:,0:4])
		return Data
		
	def NearestNeighbour(self,Data,Labels,TestPoint):
		# Calculate Nearest Neighbour
		"""
		Data		: Training Inputs
		Labels		: Training Outputs/Targets
		TestPoint	: Test Point
		"""
		Distances = self.EuclidenDistance(Data,TestPoint)				# Calculating Distance to all points from Test Point
		Neighbours = np.argsort(Distances)								# Extracting Indices in increasing order
		
		Class = list(Labels[list(Neighbours[:self.k])])					# Extracting Labels of Data
		Nearest_Class = Class[0]										# Nearest Point Label
		
		return Nearest_Class
		
	def Predict_NearestNeighbours(self,Data,Labels,TestSet):
		# Predict Nearest Neighbours for Testset
				"""
		Data		: Training Inputs
		Labels		: Training Outputs/Targets
		TestSet		: Test Points
		"""
		Predictions = []
		for test in TestSet:
			Predictions.append(self.Neighbours(Data,Labels,test))
			
		return np.array(Predictions)
		
'''
Data = np.array([[1,1,1],[2,1,2],[2,2,2],[3,3,3]])
Labels = np.array([0,1,1,2])
Test = np.array([[4,4,4]])
'''
knn = KNN(3)
Y_Pred = knn.Predict_Neighbours(X_train,Y_train,X_val)					# Predictions
Accuracy = accuracy_score(Y_val,Y_Pred)									# Calulating Accuracy
print (Accuracy)
