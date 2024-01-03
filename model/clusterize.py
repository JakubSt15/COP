from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, lfilter
import os


'''
Imports csv where each channel is a column and returns a df where
eachchannel is a row
'''
def csv_to_samples(csv_path, transpose=False):
    df = pd.read_csv(csv_path)
    if transpose: return df.T
    return df


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
        print(len(_data[0]))
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
Gets the signals from csvand returns csv with their strengths
'''
def get_signal_strength(data):
    _data = csv_to_samples(data) if type(data) is str else data
    pasma = [[8, 13], [13,30]]
    Fs=512
    moc_pasmowa = []
    for i in range(19):
        strengths_for_band = []
        for j in range(len(pasma)):
            print(i)
            signal = _data.iloc[i, :]
            (a, b) = butter(4, [pasma[j][0]/(Fs/2), pasma[j][1]/(Fs/2)], btype='band')
            filtered = lfilter(a, b, signal)
            moc = np.mean(list(map(lambda x:pow(x,2),filtered)))
            strengths_for_band.append(moc)
        moc_pasmowa.append(strengths_for_band)
    df = pd.DataFrame(moc_pasmowa)
    df.to_csv(f'./model/moce/moc_{count_files("./model/moce")}.csv', index=False)

'''
Gets data as path or ready data and returns clusterized labels
labels differnetiate attack from no attacked channels during overall attack
'''
def get_kmeans_labels(data):
    _data = csv_to_samples(data) if type(data) is str else data
    print(KMeans(n_clusters=2).fit_predict(_data))
    return KMeans(n_clusters=2).fit_predict(_data)

'''
Creates a mask of the moment of attack, it should be then pasted into its place in the big mask
'''
def create_mask_of_attack(data):
    labels = np.array(get_kmeans_labels(data))
    mask = np.tile(labels, (35840, 1))
    df = pd.DataFrame(mask)
    df.to_csv('output.csv', index=False)


'''
count number of files in the directory
'''
def count_files(dir):
    count = 0
    dir_path = fr'{dir}'
    for path in os.scandir(dir_path):
        if path.is_file():
            count += 1
    return count


# visualize_kmeans("./model/moce/moc_0.csv")
get_signal_strength("./model/train_data/_1.csv")
print(create_mask_of_attack("./model/moce/moc_0.csv"))