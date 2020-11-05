# Principal Component Analysis (PCA)
# Source: https://www.geeksforgeeks.org/ml-principal-component-analysispca/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mclrs
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import scipy
from mpl_toolkits import mplot3d
my_cmap = mclrs.LinearSegmentedColormap.from_list("", ["red","gold","purple"])

# Principal Component Analysis
class PCA():
    def __init__(self,X):
        self.X = X - np.mean(X,axis=1,keepdims=True)
        self.N = X.shape[1]
        self.Cxx = self.CoVarianceMatrix()
        self.Q = self.EigenVectors()
        self.P = self.ProjectionMatrix()
    
    def CoVarianceMatrix(self):
        O = np.dot(self.X,self.X.T)
        return O/(self.N - 1)
    
    def EigenVectors(self):
        EigenValues, EigenVectors = np.linalg.eig(self.Cxx)
        return EigenVectors
    
    def ProjectionMatrix(self):
        P = np.transpose(self.Q)
        return P
    
    def fit(self):
        Y = np.dot(self.P,self.X).real
        return Y
    
    def Verify(self,Y):
        # Checks if Cxx is Symmetric Matrix and Cyy is Diagonal Matrix
        print ("Cxx is Symmetric Matrix:",np.all(self.Cxx == self.Cxx.T))
        print ("Cxx = \n", self.Cxx)
        print()
        Cyy = np.round((np.dot(Y,Y.T).real)/(self.N - 1), decimals=4)
        print ("Cyy is Diagonal Matrix and Y is Decorrelated:", np.all(Cyy == np.diag(np.diagonal(Cyy))))
        print ("Cyy = \n", Cyy)

# Importing Data
Dataset = load_wine()
X = Dataset['data']
y = Dataset['target']
Target_Names = Dataset['target_names']
print (X.shape)
print (y.shape)
print (Target_Names)

# Standardising Data
Standardiser = StandardScaler()
Standardiser.fit(X)
X = Standardiser.transform(X)

# Extracting Principal Components
pca = PCA(n_components = 3)
pca.fit(X)
X_PCA = pca.transform(X)

# Visualising Data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("PCA Projections")
ax.scatter(X_PCA[:,0], X_PCA[:,1], X_PCA[:,2],c=y,cmap=my_cmap)
plt.show()
