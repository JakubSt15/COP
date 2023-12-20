import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense,TimeDistributed
from keras_tuner import RandomSearch
import tensorflow as tf
import mne
import gc


def build_model(hp):
    model = Sequential([
        #TimeDistributed(Dense(hp.Int('units', min_value=10, max_value=50, step=5)),input_shape=(frame_size, 19)),
        LSTM(units=hp.Int('lstm_units', min_value=1, max_value=5, step=2),return_sequences=True,activation=hp.Choice("activation", ["relu", "tanh"]),input_shape=(1000, 19)),
        Dense(units=19)
    ])
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1, sampling="log")
    model.compile( tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def cut_into_windows_edf(edf):
    raw_data=mne.make_fixed_length_epochs(edf,2)
    raw_data=raw_data.get_data()
    x=tf.transpose(tf.convert_to_tensor(raw_data),perm=[0,2,1])
    data_shifted = edf.get_data()[:, 1:]
    times_shifted = edf.times[1:]
        
    info = mne.create_info(ch_names=edf.ch_names, sfreq=edf.info['sfreq'], ch_types='eeg')
    edf = mne.io.RawArray(data_shifted, info, first_samp=edf.first_samp + 1)   
    raw_data=mne.make_fixed_length_epochs(edf,2)
    raw_data=raw_data.get_data()
    y=tf.transpose(tf.convert_to_tensor(raw_data),perm=[0,2,1])
    
    length=raw_data.shape[0]-1
    x=x[:y.shape[0]-1,:,:19]
    y=y[:-1,:,:19]
    del raw_data
    (x, y) = normalize(x, y)

    print(x[0])
    print("X len: ", len(x))
    print("X[0] len: ",len(x[0]))
    return  x, y, length

def cut_into_windows_csv(csv, hz=500):
    sec = 2
    length_of_window = int(sec * hz)
    data_array = np.array(csv)
    num_windows = len(data_array) // length_of_window
    windows = [data_array[i * length_of_window:(i + 1) * length_of_window] for i in range(num_windows)]
    tensor_windows = [np.array(window) for window in windows]
    print(len(windows))


    length=len(tensor_windows[0])-1
    x = np.array(tensor_windows)  # Convert to NumPy array
    y = np.array(tensor_windows)
    x=x[:y.shape[0]-1,:,:19]
    y=y[:-1,:,:19]
    del raw_data
    (x, y) = normalize(x, y)
    return  x, y, length

def new_model_train_sequential(train_data_files, units=1, epochs=1, batch_size=1000, frame_size=1000, save_file="./model/model_saved/"):
    split_ratio = 0.8

    model_checkpoint=ModelCheckpoint(
        filepath=f"{save_file}checkpoint",
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='auto'
    )
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=2,
        directory='model/model_saved/tuner_results',
        project_name='hyperparameter_tuning')
    earlyStopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    edf=mne.io.read_raw_edf(train_data_files[0],preload=True)

    (x, y, length) = cut_into_windows_edf(edf)
    return 0
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    split = int(split_ratio * length)
    train_dataset = dataset.take(split).batch(batch_size)
    val_dataset = dataset.skip(split).batch(batch_size)
    gc.collect()
    tuner.search(train_dataset, batch_size=batch_size, epochs=epochs,use_multiprocessing=True,callbacks=[model_checkpoint,earlyStopping],validation_data=val_dataset)

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    for file in train_data_files:
        edf=mne.io.read_raw_edf(file,preload=True)
        raw_data=mne.make_fixed_length_epochs(edf,frame_size/500)
        raw_data=raw_data.get_data()

        x=tf.transpose(tf.convert_to_tensor(raw_data),perm=[0,2,1])
            
        data_shifted = edf.get_data()[:, 1:]
        times_shifted = edf.times[1:]
        info = mne.create_info(ch_names=edf.ch_names, sfreq=edf.info['sfreq'], ch_types='eeg')
        
        edf = mne.io.RawArray(data_shifted, info, first_samp=edf.first_samp + 1)   
        raw_data=mne.make_fixed_length_epochs(edf,frame_size/500)
        raw_data=raw_data.get_data()
        y=tf.transpose(tf.convert_to_tensor(raw_data),perm=[0,2,1])
        
        length=raw_data.shape[0]-1
        x=x[:y.shape[0]-1,:,:19]
        y=y[:-1,:,:19]
        (x, y) = normalize(x, y)
        
        dataset = tf.data.Dataset.from_tensor_slices((x,y))
        del raw_data
        split = int(split_ratio * length)
        train_dataset = dataset.take(split).batch(batch_size)
        val_dataset = dataset.skip(split).batch(batch_size)
        
        gc.collect()
        model.fit(train_dataset, batch_size=batch_size, epochs=epochs,use_multiprocessing=True,callbacks=[model_checkpoint],validation_data=val_dataset)

    model.save(save_file)

def new_model_train_classifier(train_data_files, units=1, epochs=1, batch_size=1000, frame_size=1000, save_file="./model/model_saved/"):
    split_ratio = 0.8

    model_checkpoint=ModelCheckpoint(
        filepath=f"{save_file}checkpoint",
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='auto'
    )
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=2,
        directory='model/model_saved/tuner_results',
        project_name='hyperparameter_tuning')
    earlyStopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    csv = np.genfromtxt(train_data_files[0], delimiter=",", dtype=int, names=True)

    (x, y, length) = cut_into_windows_csv(csv)
    return 
    dataset = tf.data.Dataset.from_tensor_slices((x,y))
    split = int(split_ratio * length)
    train_dataset = dataset.take(split).batch(batch_size)
    val_dataset = dataset.skip(split).batch(batch_size)
    gc.collect()
    tuner.search(train_dataset, batch_size=batch_size, epochs=epochs,use_multiprocessing=True,callbacks=[model_checkpoint,earlyStopping],validation_data=val_dataset)

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    for file in train_data_files:
        edf=mne.io.read_raw_edf(file,preload=True)
        raw_data=mne.make_fixed_length_epochs(edf,frame_size/500)
        raw_data=raw_data.get_data()

        x=tf.transpose(tf.convert_to_tensor(raw_data),perm=[0,2,1])
            
        data_shifted = edf.get_data()[:, 1:]
        times_shifted = edf.times[1:]
        info = mne.create_info(ch_names=edf.ch_names, sfreq=edf.info['sfreq'], ch_types='eeg')
        
        edf = mne.io.RawArray(data_shifted, info, first_samp=edf.first_samp + 1)   
        raw_data=mne.make_fixed_length_epochs(edf,frame_size/500)
        raw_data=raw_data.get_data()
        y=tf.transpose(tf.convert_to_tensor(raw_data),perm=[0,2,1])
        
        length=raw_data.shape[0]-1
        x=x[:y.shape[0]-1,:,:19]
        y=y[:-1,:,:19]
        (x, y) = normalize(x, y)
        
        dataset = tf.data.Dataset.from_tensor_slices((x,y))
        del raw_data
        split = int(split_ratio * length)
        train_dataset = dataset.take(split).batch(batch_size)
        val_dataset = dataset.skip(split).batch(batch_size)
        
        gc.collect()
        model.fit(train_dataset, batch_size=batch_size, epochs=epochs,use_multiprocessing=True,callbacks=[model_checkpoint],validation_data=val_dataset)

    model.save(save_file)

def normalize(x, y):
    min_val = np.min(x)
    max_val = np.max(x)
    _x = (x - min_val) / (max_val - min_val)
    _y = (y - min_val) / (max_val - min_val)
    return (_x, _y)


# new_model_train_classifier(["model/train_data/mask_p11_Record1.csv"])
new_model_train_sequential(["model/train_data/p11_Record1.edf"])