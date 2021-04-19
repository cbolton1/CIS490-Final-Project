import pandas as pd
import matplotlib as plt
from sklearn.cluster import KMeans

#Allow display of all columns for debugging
pd.set_option('display.max_columns',None)

#Read the dataset into a dataframe, remove the date
df = pd.read_csv("weatherAUS.csv")
df = df.drop(columns=['Date'],inplace=False)

print(df.head(5))

#Create dummy columns for categorical data
data_norm = df.copy()
data_norm = pd.get_dummies(data_norm, columns=["Location","WindGustDir","WindDir9am","WindDir3pm","RainToday","RainTomorrow"])

print(data_norm.head(5))

#K-Means Clustering
kmeans = KMeans(3)
clusters = kmeans.fit_predict(data_norm)
labels = pd.DataFrame(clusters)


