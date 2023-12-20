from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

'''
Imports csv where each channel is a column and returns a df where
eachchannel is a row
'''
def csv_to_samples(csv_path):
    df = pd.read_csv(csv_path)
    return df.T


'''
Uses PCA to get only 3 principal components to visualize with labels
taken from KMeans
'''
def visualize_kmeans(data):
    _data = data

    # If data is a csv path, operate on csv, else use it as ready data
    if type(data) is str:
        _data = csv_to_samples(data)

    scaler = MinMaxScaler()
    _data = scaler.fit_transform(_data)

    n = 2
    if n == 3:
        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(_data)
        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(data_pca)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=labels, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('KMeans Clustering Visualization')

        plt.show()
    elif n == 2:
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(_data)
        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(data_pca)

        # Create a 2D scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('KMeans Clustering Visualization (2D)')

        plt.show()



'''
Gets data as path or ready data and returns clusterized labels
labels differnetiate attack from no attacked channels during overall attack
'''
def get_kmeans_labels(data):
    _data = csv_to_samples(data) if type(data) is str else data
    return KMeans(n_clusters=2).fit_predict(_data)

'''
Creates a mask of the moment of attack, it should be then pasted into its place in the big mask
'''
def create_mask_of_attack(data):
    labels = np.array(get_kmeans_labels(data))
    mask = np.tile(labels, (100, 1))
    df = pd.DataFrame(mask)
    df.to_csv('output.csv', index=False)


visualize_kmeans("./model/train_data/_1.csv")
# print(create_mask_of_attack("./model/train_data/_1.csv"))