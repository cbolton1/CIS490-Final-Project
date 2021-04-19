import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns

#Allow display of all columns for debugging
pd.set_option('display.max_columns',None)

#Read the dataset into a dataframe, remove most of the qualitative data
df = pd.read_csv("weatherAUS.csv")
df = df.drop(columns=['Date','Location','WindGustDir','WindDir9am','WindDir3pm'],inplace=False)

print(df.head(5))

#Create dummy columns for categorical data
data_norm = df.copy()
data_norm = pd.get_dummies(data_norm, columns=["RainToday","RainTomorrow"])

#Fill NaN values with means
data_norm = data_norm.where(pd.notna(data_norm),data_norm.mean(),axis="columns")

print(data_norm.head(5))

test_set = data_norm.head(5000) #how much of the dataset is being used in order to save time during testing

#K-Means Clustering
x = test_set.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
kmeans = KMeans(n_clusters = 3) #number of clusters being used
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)
kmeans.cluster_centers_

#Test cluster numbers for error 1 through 13
Error =[]
for i in range(1, 13):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)

#Plot elbow method results
plt.plot(range(1, 13), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()

#Plot kmeans results in scatterplot
plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1],s=20,marker='o',c=y_kmeans,cmap='rainbow')
plt.show()