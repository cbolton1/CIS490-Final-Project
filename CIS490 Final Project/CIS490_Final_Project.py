import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from gap_statistic import OptimalK

# Allow display of all columns for debugging
pd.set_option('display.max_columns', None)

# Read the dataset into a dataframe, remove most of the qualitative data
df = pd.read_csv("weatherAUS.csv")
df = df.drop(columns=['Date', 'Location', 'WindGustDir',
                      'WindDir9am', 'WindDir3pm'], inplace=False)

# print(df.head(5))

# Create dummy columns for categorical data
data_norm = df.copy()
data_norm = pd.get_dummies(data_norm, columns=["RainToday", "RainTomorrow"])

# Fill NaN values with means and normalize the data
data_norm = data_norm.where(
    pd.notna(data_norm), data_norm.mean(), axis="columns")
col_maxes = data_norm.max()
df_max = col_maxes.max()
data_norm = (data_norm/df_max)

# print(data_norm.head(5))


# how much of the dataset is being used in order to save time during testing
test_set = data_norm.head(5000)


# # K-Means Clustering (4 features, mintemp, maxtemp, rainfall, evaporation)
# x = test_set.iloc[:, [1, 2, 3, 4]].values
# # number of clusters being used
# kmeans = KMeans(init='random', n_init=10, n_clusters=3)
# y_kmeans = kmeans.fit_predict(x)
# print(y_kmeans)
# print(kmeans.cluster_centers_)

# # Test cluster numbers for error 1 through kmax

# kmax = 15

# Error = []
# for i in range(2, kmax+1):
#     kmeans = KMeans(n_clusters=i).fit(x)
#     kmeans.fit(x)
#     Error.append(kmeans.inertia_)

# # Plot elbow method results
# plt.plot(range(2, kmax+1), Error)
# plt.title('Elbow method')
# plt.xlabel('No of clusters')
# plt.ylabel('Error')
# plt.show()

# # Silhouette test
# sil = []
# for k in range(2, kmax+1):
#     kmeans = KMeans(n_clusters=k).fit(x)
#     labels = kmeans.labels_
#     sil.append(silhouette_score(x, labels, metric='euclidean'))

# plt.plot(range(2, kmax+1), sil)
# plt.title('Silhouette Method')
# plt.xlabel('No of clusters')
# plt.ylabel('Silhouette Score')
# plt.show()


# # Plot kmeans results in scatterplot
# plt.figure(figsize=(10, 10))
# plt.scatter(x[:, 0], x[:, 1], s=20, marker='o', c=y_kmeans, cmap='rainbow')
# plt.show()


# # K-Means Clustering (features Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, RainTomorrow_No, RainTomorrow_Yes)
# x = test_set.iloc[:, [8, 9, 10, 11, 18, 19]].values
# # number of clusters being used
# kmeans = KMeans(init='random', n_init=10, n_clusters=3)
# y_kmeans = kmeans.fit_predict(x)
# print(y_kmeans)
# kmeans.cluster_centers_

# # Test cluster numbers for error 1 through kmax

# kmax = 15

# Error = []
# for i in range(2, kmax+1):
#     kmeans = KMeans(n_clusters=i).fit(x)
#     kmeans.fit(x)
#     Error.append(kmeans.inertia_)

# # Plot elbow method results
# plt.plot(range(2, kmax+1), Error)
# plt.title('Elbow method')
# plt.xlabel('No of clusters')
# plt.ylabel('Error')
# plt.show()

# # Silhouette test
# sil = []
# for k in range(2, kmax+1):
#     kmeans = KMeans(n_clusters=k).fit(x)
#     labels = kmeans.labels_
#     sil.append(silhouette_score(x, labels, metric='euclidean'))

# plt.plot(range(2, kmax+1), sil)
# plt.title('Silhouette Method')
# plt.xlabel('No of clusters')
# plt.ylabel('Silhouette Score')
# plt.show()


# # Plot kmeans results in scatterplot
# plt.figure(figsize=(10, 10))
# plt.scatter(x[:, 0], x[:, 1], s=20, marker='o', c=y_kmeans, cmap='rainbow')
# plt.show()


# # K-Means Clustering (largest set)
# x = test_set.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9,
#                       10, 11, 12, 13, 14, 15, 16, 17, 18, 19]].values
# # number of clusters being used
# kmeans = KMeans(init='random', n_init=10, n_clusters=2)
# y_kmeans = kmeans.fit_predict(x)
# print(y_kmeans)
# kmeans.cluster_centers_

# # Test cluster numbers for error 1 through kmax

# kmax = 15

# Error = []
# for i in range(2, kmax+1):
#     kmeans = KMeans(n_clusters=i).fit(x)
#     kmeans.fit(x)
#     Error.append(kmeans.inertia_)

# # Plot elbow method results
# plt.plot(range(2, kmax+1), Error)
# plt.title('Elbow method')
# plt.xlabel('No of clusters')
# plt.ylabel('Error')
# plt.show()

# # Silhouette test
# sil = []
# for k in range(2, kmax+1):
#     kmeans = KMeans(n_clusters=k).fit(x)
#     labels = kmeans.labels_
#     sil.append(silhouette_score(x, labels, metric='euclidean'))

# plt.plot(range(2, kmax+1), sil)
# plt.title('Silhouette Method')
# plt.xlabel('No of clusters')
# plt.ylabel('Silhouette Score')
# plt.show()


# # Plot kmeans results in scatterplot
# plt.figure(figsize=(10, 10))
# plt.scatter(x[:, 0], x[:, 1], s=20, marker='o', c=y_kmeans, cmap='rainbow')
# plt.show()


# Hierarchical testing - Agglomerative

# Method to draw dendogram - https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_silhouette(x, kmax):
    # Silhouette test
    sil = []
    for k in range(2, kmax+1):
        agg = AgglomerativeClustering(n_clusters=k, linkage='single')
        y = agg.fit_predict(x)
        sil.append(silhouette_score(x, y, metric='euclidean'))

    plt.plot(range(2, kmax+1), sil)
    plt.title('Silhouette Method')
    plt.xlabel('No of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()


# Variable set 1 (4 features, mintemp, maxtemp, rainfall, evaporation)
x = test_set.iloc[:, [1, 2, 3, 4]].values
agg = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage='single')
y_agg = agg.fit_predict(x)

plt.title('Agglomerative Clustering Dendrogram')
# plot the top levels of the dendrogram
plot_dendrogram(agg, truncate_mode='level')
plt.xlabel("Number of points in node")
plt.show()


# Plot the silhouette scores
plot_silhouette(x, 9)

# Plot kmeans results in scatterplot
plt.figure(figsize=(10, 10))
agg = AgglomerativeClustering(n_clusters=5, linkage='single')
y_agg = agg.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s=20, marker='o', c=y_agg, cmap='rainbow')
plt.show()


# # Variable set 2 (features Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, RainTomorrow_No, RainTomorrow_Yes)
x = test_set.iloc[:, [8, 9, 10, 11, 18, 19]].values
agg = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage='single')
y_agg = agg.fit_predict(x)

plt.title('Agglomerative Clustering Dendrogram')
# plot the top levels of the dendrogram
plot_dendrogram(agg, truncate_mode='level')
plt.xlabel("Number of points in node")
plt.show()


# Plot the silhouette scores
plot_silhouette(x, 9)

# Plot kmeans results in scatterplot
plt.figure(figsize=(10, 10))
agg = AgglomerativeClustering(n_clusters=5, linkage='single')
y_agg = agg.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s=20, marker='o', c=y_agg, cmap='rainbow')
plt.show()

# # Variable set 3 (largest set)
x = test_set.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]].values
agg = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage='single')
y_agg = agg.fit_predict(x)

plt.title('Agglomerative Clustering Dendrogram')
# plot the top levels of the dendrogram
plot_dendrogram(agg, truncate_mode='level')
plt.xlabel("Number of points in node")
plt.show()


# Plot the silhouette scores
plot_silhouette(x, 9)

# Plot kmeans results in scatterplot
plt.figure(figsize=(10, 10))
agg = AgglomerativeClustering(n_clusters=5, linkage='single')
y_agg = agg.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], s=20, marker='o', c=y_agg, cmap='rainbow')
plt.show()