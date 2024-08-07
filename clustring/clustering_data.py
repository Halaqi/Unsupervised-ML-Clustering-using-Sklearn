import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv(r"D:\programming\Machine learning\clustring\Mall_Customers.csv")
x_train = data.iloc[:, [3, 4]].values

# Finding the proper number of clusters
wcss_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x_train)
    wcss_list.append(kmeans.inertia_)

# Using the elbow method to display the number of clusters 
plt.plot(range(1, 11), wcss_list)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Training the data with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(x_train)

# Visualizing the data before clusting
plt.scatter(x_train[: , 0], x_train[:, 1], s=100, c='blue', label='Cluster')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Visualizing the data of clusters
plt.scatter(x_train[y_pred == 0, 0], x_train[y_pred == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x_train[y_pred == 1, 0], x_train[y_pred == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x_train[y_pred == 2, 0], x_train[y_pred == 2, 1], s=100, c='yellow', label='Cluster 3')
plt.scatter(x_train[y_pred == 3, 0], x_train[y_pred == 3, 1], s=100, c='green', label='Cluster 4')
plt.scatter(x_train[y_pred == 4, 0], x_train[y_pred == 4, 1], s=100, c='cyan', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='magenta', label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
