import pandas as pd
from sklearn.datasets import load_iris
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = { 0:'setosa',1:'versicolor', 2:'virginica'}
df['target'] = df['target'].map(target_names)
print("Original Data")
print("\n-----------------------------------------------------------------")
print(df.head())
print("\n-----------------------------------------------------------------")
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

x = df.drop(['target'],axis=1)
y = df['target']
print(x.head())
print("\n-----------------------------------------------------------------")

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(finalDf.head())
print("\n-----------------------------------------------------------------")

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['setosa', 'versicolor', 'virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()


