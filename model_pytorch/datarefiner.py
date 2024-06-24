import numpy as np
import scipy
import tensorflow as tf
from scipy import signal, stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class DataRefiner:
    def __init__(self):
        self.low_cut = 3
        self.high_cut = 13
        self.frequency = 512

    def __butter_filter(self, raw_data):
        b, a = signal.butter(N=4, Wn=[self.low_cut/(self.frequency/2), self.high_cut/(self.frequency/2)], btype='bandpass')
        for channel in range(19):
            raw_data[:, channel] = signal.lfilter(b, a, raw_data[:, channel])
        return raw_data
    
    def refine(self, array_2d, visualize=False):   
        if visualize:
            fig, axs = plt.subplots(7, 1, figsize=(10, 18))

        array_2d = self.__remove_outliers_and_fill(array_2d)

        array_2d = self.__butter_filter(array_2d)

        array_2d = self.power(array_2d)

        array_2d = self.__moving_average(array_2d, 1500)

        array_2d = self.__decimation(array_2d, 128)

        array_2d = self.quantize_power(array_2d.T, 1000)

        array_2d = self.__transformation(array_2d.T)
        return array_2d

    def power(self, attack):
        return attack ** 2


    def __moving_average(self, data, window_size=2):
        data = data.T
        num_channels, num_samples = data.shape
        if window_size > num_samples:
            window_size = num_samples
        cumsum = np.cumsum(data, axis=1)
        moving_averages = (cumsum[:, window_size:] - cumsum[:, :-window_size]) / window_size
        return np.hstack((data[:, :window_size-1], moving_averages)).T

    def __decimation(self, data, n=2):
        return data[::n]

    def __transformation(self, data):
        data = data.T
        scaler = MinMaxScaler((-1, 1))
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
        
    def quantize_power(self, data, num_levels=2000):
        quantization_bins = np.linspace(np.min(data), np.max(data), num=num_levels)
        quantized_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            quantized_data[:, i] = np.digitize(data[:, i], quantization_bins)
        return quantized_data.T
