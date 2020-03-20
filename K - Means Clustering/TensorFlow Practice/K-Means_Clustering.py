"""
Implementation of K-Means Clustering Algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
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
np.random.seed(200)

print ("--------------------------------------------Imported Libraries---------------------------------")


# Setting cmap colors
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","green","blue","gold","purple","black"])
np.random.seed(0)

# Generating Dataset
Centres = np.random.uniform(low=210,high=255,size=(6,3))
Data, labels = make_blobs(n_samples=1500, n_features=3, centers=Centres, cluster_std=1.5,random_state=1)
print(Data.shape, labels.shape)

# Visualising Data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Input Data")
ax.scatter(Data[:,0], Data[:,1], Data[:,2],c=labels,cmap=my_cmap)
plt.show()

print ("-------------------------------------Data Imported and Visualised-------------------------------")


class KMeans():
	
	def __init__(self,k):
		self.k = k
	
	def EuclidenDistance(self,X,y):
		# Calculate Eucliden Distance
		d = np.linalg.norm(X-y,axis=1)
		return d
		
	def InitiliseCentroids(self,Data_Labelled):
		# Initilising Centroids of Clusters as Random Points and Changing their Labels
		m = Data_Labelled.shape[0]
		r = np.random.randint(0,m,size=(self.k,))
		Centroids = np.zeros((self.k,Data_Labelled.shape[1]-1))
		label = 0
		idx = 0
		for i in r:
			Data_Labelled[i][3] = label
			Centroids[idx] = Data_Labelled[i,0:3]
			idx += 1
			label += 1
		
		return Centroids
	
	def VarianceLoss(self,Data_Labelled):
		# Variance Loss for K-Means Clustering 
		Loss = 0
		for i in range(self.k):
			x = np.array([Data_Labelled[j,0:3] for j in range(Data_Labelled.shape[0]) if Data_Labelled[j][3] == i])
			Loss += np.std(np.std(x,axis=0))

		return Loss
		
	def CalculateCentroid(self,Data_Labelled,label):
		# Calculating Centroid
		return np.mean(np.array([Data_Labelled[j,0:3] for j in range(Data_Labelled.shape[0]) if Data_Labelled[j][3] == label]),axis=0)
		
		
	def VisualiseData(self,Data,Title):
		# Visualising Data
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_title(Title)
		ax.scatter(Data[:,0], Data[:,1], Data[:,2],c=Data[:,3],cmap=my_cmap)
		plt.show()
		
	def ClusterData(self,Data,Epochs):
		# Cluster Data
		
		for e in range(Epochs):
			
			# Partial Output
			Labels = np.ones((Data.shape[0],1)) * (self.k + 1)
			Data_Labelled = np.append(Data,Labels,axis=1)
		
			# Initialising Centroids
			Centroids = self.InitiliseCentroids(Data_Labelled)
			
			# Iteration over the Dataset
			for i in range(Data.shape[0]):
				
				# Label of the Nearest Neighbour
				label = np.argmin(self.EuclidenDistance(Centroids,Data_Labelled[i,0:3]))
				Data_Labelled[i][3] = label
				
				# Recalculating Centroids
				Centroids[label] = self.CalculateCentroid(Data_Labelled,label)
			
			# Visualising Data
			Title = "Epoch:"+str(e)
			self.VisualiseData(Data_Labelled,Title)
			
			if e==0:
				Loss = self.VarianceLoss(Data_Labelled)
				Output = Data_Labelled
			else:
				if self.VarianceLoss(Data_Labelled) < Loss:
					Output = Data_Labelled
					Loss = self.VarianceLoss(Data_Labelled)
		
		return Output,Loss
		
KMeans = KMeans(6)
Clustered_Data = KMeans.ClusterData(Data,3)[0]
KMeans.VisualiseData(Clustered_Data,"Final Output")
