# Learning t-SNE
# Source: https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import cauchy, norm, t
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Student's t Distribution with Degree of Freedom = 1 and Normal/Gaussian Distrubution are used for Measuring Similarities
x = np.linspace(-10,10,500)
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","green","blue","gold","purple","black"])

# Degree of Freedom
df = 1

n_mean, n_var, n_skew, n_kurt = norm.stats(moments='mvsk')
c_mean, c_var, c_skew, c_kurt = t.stats(df,moments='mvsk')

plt.plot(x,norm.pdf(x),'r',label='Normal Distribution')
plt.plot(x,cauchy.pdf(x),'b',label="Student's t Distribution")
plt.grid()
plt.legend()
plt.title('Normal Distribution vs Student t-Distribution')
plt.savefig('Images/SimilarityDistributions.png')
plt.show()


# Loading Dataset
Train = pd.read_csv('Dataset/train.csv')

X_Train = ((Train.loc[:, Train.columns != 'label']).to_numpy())
Y_Train = ((Train['label']).to_numpy())

X_Train, _, Y_Train, _ = train_test_split(X_Train, Y_Train, train_size = 10000, random_state=42, stratify=Y_Train)

print ('Train Data Shape:',X_Train.shape)


# Applying PCA
pca = PCA(n_components = 2)
pca.fit(X_Train)
X_PCA = pca.transform(X_Train)
X_PCA = np.append(X_PCA, np.expand_dims(Y_Train,axis=1), axis=1)
X_PCA = pd.DataFrame(data=X_PCA, columns=['Component-1', 'Component-2', 'Label'])

# Visualising Data
ax = sns.lmplot(x='Component-1',
           y='Component-2',
           data=X_PCA,
           fit_reg=False,
           hue = 'Label',
           height = 7.5,
           aspect = 1.5,
           legend=True)
plt.xlabel('PCA Component-1')
plt.ylabel('PCA Component-2')
plt.savefig('Images/PCA_Results.png')
plt.show()


# Applying t-SNE
tsne = TSNE(n_components=2)
X_TSNE = tsne.fit_transform(X_Train)
X_TSNE = np.append(X_TSNE, np.expand_dims(Y_Train,axis=1), axis=1)
X_TSNE = pd.DataFrame(data=X_TSNE, columns=['Component-1', 'Component-2', 'Label'])

# Visualising Data
ax = sns.lmplot(x='Component-1',
           y='Component-2',
           data=X_TSNE,
           fit_reg=False,
           hue = 'Label',
           height = 7.5,
           aspect = 1.5,
           legend=True)
plt.xlabel('TSNE Component-1')
plt.ylabel('TSNE Component-2')
plt.savefig('Images/TSNE_Results.png')
plt.show()
