#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import ListedColormap


# In[2]:


from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#read the df
df = pd.read_csv('binclass.txt') 
df.columns = ["positive", "negative", "y"]
x_d = df[['positive', 'negative']]
#The fit(df) method is used to compute the mean and std dev for a given feature to be used further for scaling
x_d = StandardScaler().fit_transform(x_d)
y = df['y'] 
#gaussian Naive Bayes (GNB) is a classification technique used in Machine Learning (ML) based on the probabilistic approach and Gaussian distribution
mod=GaussianNB()
mod.fit(x_d,y)
yhat=mod.predict(x_d)
print('Accuracy Score: ', mod.score(x_d, y))


# In[3]:


df


# In[4]:


import seaborn as sns
sns.scatterplot(x="positive", y="negative",palette=['r','b'] , hue='y', data=df)


# In[5]:


from matplotlib.colors import ListedColormap
X_set, y_set = x_d, y
#The numpy. meshgrid function is used to create a rectangular grid out of two given one-dimensional arrays representing the Cartesian indexing or Matrix indexing. 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
#draw contour lines and filled contours,
plt.contourf(X1, X2, mod.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('white', 'black')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(( 'blue','red'))(i), label = j,marker='+')
plt.title('Naive Bayes Classification')
plt.xlabel('positive')
plt.ylabel('negative')
plt.legend()
plt.show()


# In[6]:


#importing the second data
data = pd.read_csv('binclassv2.txt')
data.columns = ["positive", "negative", "y"]
x_d = data[["positive", "negative"]]
#The fit(df) method is used to compute the mean and std dev for a given feature to be used further for scaling
x_d= StandardScaler().fit_transform(x_d)
y = data['y']
#gaussian Naive Bayes (GNB) is a classification technique used in Machine Learning (ML) based on the probabilistic approach and Gaussian distribution
mod=GaussianNB()
mod.fit(x_d,y)
yhat=mod.predict(x_d)
print('Accuracy Score: ', mod.score(x_d, y))


# In[7]:


#plotting the data
sns.scatterplot(x="positive", y="negative",palette=['blue','red'] , hue='y', data=data)


# In[8]:


from matplotlib.colors import ListedColormap
## Visualising the Training set results
X_set, y_set = x_d, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
plt.contourf(X1, X2, mod.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('white', 'black')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(( 'blue','red'))(i), label = j,marker='+')
#plotting the data
plt.title('Naive Bayes Classification')
plt.xlabel('positive')
plt.ylabel('negative')
plt.legend()
plt.show()

