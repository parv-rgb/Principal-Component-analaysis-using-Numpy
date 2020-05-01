# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:15:54 2020

@author: parvp
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("ggplot")
plt.rcParams["figure.figsize"]=(12,8)

df= pd.read_csv("iris.csv", header=None)

df.columns=["sepal_length","sepal_width","petal_length","petal_width","species"]

# to drop the row which have null value

df.dropna(how="all",inplace=True)

df.head()
df.info()

# visulazie the data 

sns.scatterplot(x= df["sepal_length"],y=df["sepal_width"],
                hue=df["species"],style=df["species"])

# Standarziation of data

x= df.iloc[:,0:4].values
y= df["species"].values

# to ensure all the features have 0 mean and unit varience

from sklearn.preprocessing import StandardScaler
x= StandardScaler().fit_transform(x)

# to find covariance matrix

covariance_matrix= np.cov(x.T)

# find eigen values and eigen vectors using eigen decomposition method

eigen_values, eigen_vectors= np.linalg.eig(covariance_matrix)

"""print(eigen_values)
print(eigen_vectors)"""

# to know the variance explained by each eigen value

varaiance= [(i/sum(eigen_values))*100 for i in eigen_values]
"""print(varaiance)"""

cumulative_varaince= np.cumsum(varaiance)
"""print(cumulative_varaince)"""

sns.lineplot(x=[1,2,3,4], y= cumulative_varaince)
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.title("Explained varaince VS Number of components")
plt.show()

# we have achecived 95 variance in two eigen values so we will take only two columns of eigen vectors

# project data onto lower-dimensional linear subspace

projection_matrix = (eigen_vectors.T)[:2].T

"""print(projection_matrix)"""

x_pca= x.dot(projection_matrix)

for species in ('Iris-setosa', 'Iris-versicolor','Iris-verginica'):
    sns.scatterplot(x_pca[y==species,0],
                    x_pca[y== species,1])






























