import numpy as np
import scipy
import tensorflow as tf
from scipy import signal, stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  MinMaxScaler
import matplotlib.pyplot as  plt
class DataRefiner:
    def __init__(self):
        self.low_cut = 3
        self.high_cut = 13
        self.frequency = 512

    def __butter_filter(self, raw_data):
        b, a = signal.butter(N=4, Wn=[self.low_cut/(self.frequency/2), self.high_cut/(self.frequency/2)], btype='bandpass')
        for channel in range(19):
            raw_data[:,channel] = signal.lfilter(b, a, raw_data[:,channel])
        return raw_data

    def refine(self, array_2d):
        #axs = plt.subplots(6, 1)[1]
        array_2d = self.__remove_outliers_and_fill(array_2d)
        #axs[0].plot(array_2d[:,0])
        #axs[0].set_title('outliers')
    
        array_2d = self.__butter_filter(array_2d)
        #axs[1].plot(array_2d[:,0])
        #axs[1].set_title('filtr')

        array_2d=self.power(array_2d)
        #axs[2].plot(array_2d[:,0])
        #axs[2].set_title('moc')

        array_2d = self.__moving_average(array_2d,2000)
        #axs[3].plot(array_2d[:,0])
        #axs[3].set_title('average')

        array_2d = self.__decimation(array_2d,16)
        #axs[4].plot(array_2d[:,0])
        #axs[4].set_title('decymacja')

        array_2d = self.__transformation(array_2d.T)
        #axs[5].plot(array_2d[0,:])
        #axs[5].set_title('MinMax scaling')
        #plt.show()
        return array_2d

    # def power(self, attack):
    #     frame_size=self.frequency
    #     shift = self.frequency
    #     frames = []
    #     power = np.zeros(19)
    #     for i in range(0, attack.shape[0] - frame_size, shift):
    #         frame = attack[i:i + frame_size, :]
    #         for channel in range(19):
    #             frame[:, channel] = np.power(frame[:, channel], 2)
    #             power[channel] = np.mean(frame[:, channel])

    #         frames.append(list(power))
    #     return np.array(frames)
    
    def power(self, attack):
        return attack**2

    def __moving_average(self, data, windows_size=2):
        data = data.T
        num_channels, num_samples = data.shape
        if windows_size > num_samples:
            windows_size = num_samples
        newData = np.empty((num_channels, num_samples - windows_size + 1))
        for i, channel in enumerate(data):
            newData[i, :] = np.convolve(channel, np.ones(windows_size), 'valid') / windows_size
        return newData.T 

    def __decimation(self,data,n=2):
        return data[::n]
    
    def __transformation(self, data):
        data = data.T
        scaler=MinMaxScaler((-1,1))
        normalized_data = scaler.fit_transform(data)
        return normalized_data.T

    def __remove_outliers_and_fill(self, tensor):
        num_channels, num_samples = tensor.shape
        z_scores = np.zeros((num_channels, num_samples))
        for channel in range(num_channels):
            z_scores[channel, :] = stats.zscore(tensor[channel, :])
        threshold = 3.0
        for channel in range(num_channels):
            outlier_indices = np.where(np.abs(z_scores[channel, :]) > threshold)[0]
            for outlier in outlier_indices:
                start_index = max(outlier - 10, 0)
                end_index = min(outlier + 10, num_samples)
                tensor[channel, outlier] = np.median(tensor[channel, start_index:end_index])
        return tensor.T
