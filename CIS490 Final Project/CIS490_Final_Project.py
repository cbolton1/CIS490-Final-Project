import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

#Allow display of all columns for debugging
pd.set_option('display.max_columns',None)

#Read the dataset into a dataframe, remove the date
df = pd.read_csv("weatherAUS.csv")
df = df.drop(columns=['Date'],inplace=False)

print(df.head(5))

#Create dummy columns for categorical data
data_norm = df.copy()
data_norm = pd.get_dummies(data_norm, columns=["Location","WindGustDir","WindDir9am","WindDir3pm","RainToday","RainTomorrow"])

#Fill NaN values with means
data_norm = data_norm.where(pd.notna(data_norm),data_norm.mean(),axis="columns")

print(data_norm.head(5))

#K-Means Clustering
kmeans = KMeans(n_clusters=3)
y = kmeans.fit_predict(data_norm)
data_norm['Cluster'] = y

print(data_norm.head())


