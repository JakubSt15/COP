
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt, decimate
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from datarefiner import DataRefiner

FRAME_SIZE = 100
ROLLINGN = 100


plt.style.use("Solarize_Light2")
def plot_signal(signal, title=None):
    plt.plot(signal)
    if title is not None:
        plt.suptitle(title)
    plt.show()    

'''
    This function takes a matrix of shape (NUMBER_OF_SAMPLES, NUMBER_OF_CHANNELS)
    and reduces it to the vector of NUMBER_OF_SAMPLES where each label is set 0
    or 1 wether 1 occured at least once on given sample
'''
def unchannel_attack_labels(labels_channeled):
    labels = []
    for l in labels_channeled:
        if l.any() == 1:
            labels.append(1)
        else:
            labels.append(0)
    return labels
        


'''
    This functions frames labels array into subparts of frame_size
    and automatically sets the label for the whole frame. So basically
    reduces array size to LABELS // frame_size 
'''
def set_framed_labels(labels, frame_size, rollingN=ROLLINGN):
    labels = labels[rollingN:]
    num_frames = len(labels) // frame_size
    labels_framed = []
    for frame_idx in range(num_frames):
        start_idx = frame_idx * frame_size
        end_idx = min((frame_idx + 1) * frame_size, len(labels))
        frame_labels = labels[start_idx:end_idx]
        labels_framed.append(max(set(frame_labels), key=frame_labels.count))

    return labels_framed


'''
    This function calculates the power of a signal in np.array
'''
def calculate_power(signal):
    fft_result = np.fft.fft(signal)
    power_spectrum = np.abs(fft_result) ** 2
    return power_spectrum
    

'''
    This function returns a list of signal power on each channel on each frame.
'''
def set_labels_signal_power_by_frames(signal, frame_size, rollingN=ROLLINGN):
    signal = signal[rollingN:]
    num_frames = len(signal) // frame_size
    power_framed = []
    for frame_idx in range(num_frames):
        start_idx = frame_idx * frame_size
        end_idx = min((frame_idx + 1) * frame_size, len(signal))
        frame_channel = signal[start_idx:end_idx]
        power_frame = []
        for channel in frame_channel.T:  # Iterate over channels (transpose to iterate over channels)
            power = np.mean(channel)
            power_frame.append(power)

        power_framed.append(power_frame)

    return power_framed


def remove_outliers_by_median(series, window_size=100):
    print("Removing outliers")
    cleaned_series = series.copy()
    N = len(series)
    for i in range(N):
        start_index = max(0, i - window_size // 2)
        end_index = min(N, i + window_size // 2 + 1)
        window = series.iloc[start_index:end_index]
        median = np.median(window)
        if np.abs(series.iloc[i]) > abs(median):
            cleaned_series.iloc[i] = median

        if i % 1000 == 0: print(f"Sample {i} / {N}")
    return cleaned_series

'''
    df - should be not transposed
    plot_verbose - If you want to plot each part of processing
'''
def process_attribute_channels(df, plot_verbose=False, rollingN=ROLLINGN):
    PROCESSED_CHANNELS = []
    for id, column in enumerate(df.T):
        if plot_verbose: plot_signal(column, title=f"Raw signal");
        
        # Filtrowanie atrybutów
        column = filtfilt(*butter(5, [3, 13], btype='band', fs=512), column )
        if plot_verbose: plot_signal(column, title=f"Butter filtered signal [3 13]");

        column = column**2 # moc sygnału

        # savgol filter
        # column = savgol_filter(column, window_length=2500, polyorder=2)
        # if plot_verbose: plot_signal(column, title=f"Savgol filtered signal");


        # Średnia krocząca
        column = (pd.DataFrame(column).rolling(rollingN).mean()).to_numpy()
        if plot_verbose: plot_signal(column, title=f"SMA processed signal");

        if plot_verbose: plot_signal(column, title=f"Upsampled signal to match labels");

        scaler = MinMaxScaler(feature_range=(-1, 1))
        column = scaler.fit_transform(column.reshape(-1, 1))

        PROCESSED_CHANNELS.append(column)

    return np.squeeze(np.array(PROCESSED_CHANNELS).T, axis=None)


'''
    data_csv is an array of lists consisting of two file paths to csv ex.
            [
            ['./path/training_record_1.csv', './mask_attack_1.csv']
            ['./path/training_record_2.csv', './mask_attack_2.csv']
            ]
    Where mask attack should be a simple mask of 0 and 1 for sample by channel with or without attack
'''
def prepare_dataset_attack_model(data_csv, shuffle=False, plot_verbose=False):
    attributes, labels = [], []
    ROLLING_N = ROLLINGN

    REFINER = DataRefiner()

    for f in data_csv:
        signal_csv, labels_csv = f[0], f[1]
        attr_df = pd.read_csv(signal_csv).to_numpy()
        labels_df = pd.read_csv(labels_csv).to_numpy()
        labels_unframed = unchannel_attack_labels(labels_df)
        labels_framed = labels_unframed[::128]    
        # labels_framed = set_framed_labels(labels_unframed, FRAME_SIZE, rollingN=ROLLING_N)   # shape (samples) -> (samples//FRAME_SIZE)
        
        print("INPUT SHAPE: ", attr_df.shape)
        power_attributes = REFINER.refine(attr_df.T)
        print("OUTPUTSHAPE: ", power_attributes.shape)
        # PROCESSED_CHANNELS = process_attribute_channels(attr_df, plot_verbose=plot_verbose, rollingN=ROLLING_N)
        # power_attributes = set_labels_signal_power_by_frames(PROCESSED_CHANNELS, FRAME_SIZE, rollingN=ROLLING_N)
        attributes += power_attributes.T.tolist()
        labels += labels_framed

    if shuffle:
        combined_data = list(zip(attributes, labels))
        random.shuffle(combined_data)
        attributes, labels = zip(*combined_data)
    
    print(len(attributes))
    print(len(labels))
    return attributes, labels


'''
    DF df - attributes_df
    DF df - labels_df
    INT column_to_predict -  this column will be not included as attibute
    This returns attributes and labels out from a df file in a way:
        - remove specified column from attr df and treat the rest as attributes
        - returns only specified column for labels_df
'''
# def get_one_channel_train_data(attr_df, labels_df, column_to_predict):
#     attr = attr_df.drop(attr_df.columns[column_to_predict], axis=1).values
#     labels = None
#     if labels_df != None:
#         labels = labels_df[labels_df.columns[column_to_predict]].values
#     return attr, labels

def get_one_channel_train_data(attr_df, labels_df, column_to_predict):
    attr = attr_df.drop(attr_df.columns[column_to_predict], axis=1).values
    labels = None
    if labels_df != None:
        labels = labels_df[labels_df.columns[column_to_predict]].values
    return attr, labels


def prepare_datasets_channel_attacks(data_csv, plot_verbose=False, fast=False):
    print('Preparing multi channel dataset...')
    
    all_df = []
    if isinstance(data_csv, pd.DataFrame):
        signal_csv, labels_csv = data_csv[0], data_csv[1]
        attr_df = signal_csv
        labels_df = labels_csv
        all_df.append([attr_df, labels_df])
    else:
        for f in data_csv:
            signal_csv, labels_csv = f[0], f[1]
            attr_df = pd.read_csv(signal_csv, header=None)
            labels_df = pd.read_csv(labels_csv, header=None)
            all_df.append([attr_df, labels_df])
    
    COLUMNS = all_df[0][1].shape[1]
    
    channel_datasets = []

    for i in range(COLUMNS):
        attributes, labels = [], []
        print(f'Current column: {i}')
        for id, df in enumerate(all_df):
            print(f'\tCurrent file: {id}')
            attr_df = df[0]
            labels_df = df[1]
            attributes_raw, labels_raw = get_one_channel_train_data(attr_df, labels_df, i)
            labels_framed = set_framed_labels(labels_raw.tolist(), FRAME_SIZE)   # shape (samples) -> (samples//FRAME_SIZE
            _attr = attr_df
            if not fast:
                PROCESSED_CHANNELS = process_attribute_channels(attributes_raw, plot_verbose=plot_verbose)
                _attr = np.nan_to_num(PROCESSED_CHANNELS)

            power_attributes = set_labels_signal_power_by_frames(_attr, FRAME_SIZE)
            attributes += power_attributes
            labels += labels_framed
            
            if len(attributes) != len(labels):
                print("Inconsistent number of attributes and labels: ")
                print("     Labels      : ", len(labels))
                print("     attributes  : ", len(attributes))
                print(f" Error occured on file no. {id}")
                return -1
            
        channel_datasets.append({
            'labels': labels,
            'attr': attributes,
            'column': i
            })

        
    return channel_datasets


def prepare_datasets_channel_attacks(data_csv, plot_verbose=False, fast=False):
    print('Preparing multi channel dataset...')
    
    all_df = []
    if isinstance(data_csv, pd.DataFrame):
        signal_csv, labels_csv = data_csv[0], data_csv[1]
        attr_df = signal_csv
        labels_df = labels_csv
        all_df.append([attr_df, labels_df])
    else:
        for f in data_csv:
            signal_csv, labels_csv = f[0], f[1]
            attr_df = pd.read_csv(signal_csv, header=None)
            labels_df = pd.read_csv(labels_csv, header=None)
            all_df.append([attr_df, labels_df])
    
    COLUMNS = all_df[0][1].shape[1]
    
    channel_datasets = []

    for i in range(COLUMNS):
        attributes, labels = [], []
        print(f'Current column: {i}')
        for id, df in enumerate(all_df):
            print(f'\tCurrent file: {id}')
            attr_df = df[0]
            labels_df = df[1]
            attributes_raw, labels_raw = get_one_channel_train_data(attr_df, labels_df, i)
            labels_framed = set_framed_labels(labels_raw.tolist(), FRAME_SIZE)   # shape (samples) -> (samples//FRAME_SIZE
            _attr = attr_df
            if not fast:
                PROCESSED_CHANNELS = process_attribute_channels(attributes_raw, plot_verbose=plot_verbose)
                _attr = np.nan_to_num(PROCESSED_CHANNELS)

            power_attributes = set_labels_signal_power_by_frames(_attr, FRAME_SIZE)
            attributes += power_attributes
            labels += labels_framed
            
            if len(attributes) != len(labels):
                print("Inconsistent number of attributes and labels: ")
                print("     Labels      : ", len(labels))
                print("     attributes  : ", len(attributes))
                print(f" Error occured on file no. {id}")
                return -1
            
        channel_datasets.append({
            'labels': labels,
            'attr': attributes,
            'column': i
            })

        
    return channel_datasets


def prepare_prediction_multi_channel_datasets(data_csv, plot_verbose=False, fast=False, rollingN=1500):
    print('Preparing multi channel dataset...')
    
    all_df = []
    attr_df = data_csv
    all_df.append(attr_df)
    COLUMNS = all_df[0].shape[1]
    
    channel_datasets = []

    for i in range(COLUMNS):
        attributes = []
        print(f'Current column: {i}')
        for id, df in enumerate(all_df):
            print(f'\tCurrent file: {id}')
            attr_df = df

            attributes_raw, _ = get_one_channel_train_data(attr_df, None, i)
            _attr = attr_df

            if not fast:
                PROCESSED_CHANNELS = process_attribute_channels(attributes_raw, plot_verbose=plot_verbose, rollingN=rollingN)
                _attr = np.nan_to_num(PROCESSED_CHANNELS)

            power_attributes = set_labels_signal_power_by_frames(_attr, FRAME_SIZE)
            attributes += power_attributes
        
            
        channel_datasets.append({
            'attr': attributes,
            'column': i
            })

        
    return channel_datasets


def get_attack_sample_from_predictions(predictions, FRAME_SIZE=1000):
    looking_for_start = True
    looking_for_end = False
    start_sample = None
    end_sample = None
    for id, pred in enumerate(predictions):
        if pred == 1 and looking_for_start:
            start_sample = id * FRAME_SIZE
            looking_for_start = False
            looking_for_end = True
        if pred == 0 and looking_for_end:
            end_sample = id * FRAME_SIZE
            break
    
    return start_sample, end_sample

