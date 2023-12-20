import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense,TimeDistributed,Normalization
import tensorflow as tf
import mne
import gc

train_files=["model/train_data/p11_Record1.edf"]
channels=['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
test_file= 'model/train_data/p11_Record1.edf'
frame_size=1000
split_ratio=0.8
checkpoint_filepath="./checkpoint"

number_of_epochs=1
batch_size=1000 

from keras_tuner import RandomSearch
def build_model(hp):
    model = Sequential([
        #TimeDistributed(Dense(hp.Int('units', min_value=10, max_value=50, step=5)),input_shape=(frame_size, 19)),
        LSTM(units=hp.Int('lstm_units', min_value=10, max_value=100, step=5),return_sequences=True,activation=hp.Choice("activation", ["relu", "tanh"]),input_shape=(frame_size, 19)),
        Dense(units=19)
    ])
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1, sampling="log")
    model.compile( tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model
model_checkpoint=ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='auto'
)
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,
    directory='tuner_results',
    project_name='hyperparameter_tuning')
earlyStopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


edf=mne.io.read_raw_edf(test_file,preload=True)
raw_data=mne.make_fixed_length_epochs(edf,2)
raw_data=raw_data.get_data()
x=tf.transpose(tf.convert_to_tensor(raw_data),perm=[0,2,1])
print(x)
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

min_val = np.min(x)
max_val = np.max(x)
x = (x - min_val) / (max_val - min_val)
y = (x - min_val) / (max_val - min_val)

print(x)
# dataset = tf.data.Dataset.from_tensor_slices((x,y))
# del raw_data
# split = int(split_ratio * length)
# train_dataset = dataset.take(split).batch(batch_size)
# val_dataset = dataset.skip(split).batch(batch_size)
# gc.collect()
# tuner.search(train_dataset, batch_size=batch_size, epochs=number_of_epochs,use_multiprocessing=True,callbacks=[model_checkpoint,earlyStopping],validation_data=val_dataset)


# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
# model = tuner.hypermodel.build(best_hps)

# for file in train_files:
#     edf=mne.io.read_raw_edf(file,preload=True)
#     raw_data=mne.make_fixed_length_epochs(edf,frame_size/500)
#     raw_data=raw_data.get_data()
#     x=tf.transpose(tf.convert_to_tensor(raw_data),perm=[0,2,1])
    
#     data_shifted = edf.get_data()[:, 1:]
#     times_shifted = edf.times[1:]
    
#     info = mne.create_info(ch_names=edf.ch_names, sfreq=edf.info['sfreq'], ch_types='eeg')
#     edf = mne.io.RawArray(data_shifted, info, first_samp=edf.first_samp + 1)   
#     raw_data=mne.make_fixed_length_epochs(edf,frame_size/500)
#     raw_data=raw_data.get_data()
#     y=tf.transpose(tf.convert_to_tensor(raw_data),perm=[0,2,1])
    
#     length=raw_data.shape[0]-1
#     x=x[:y.shape[0]-1,:,:19]
#     y=y[:-1,:,:19]
#     x = (x - min_val) / (max_val - min_val)
#     y = (x - min_val) / (max_val - min_val)
    
#     dataset = tf.data.Dataset.from_tensor_slices((x,y))
#     del raw_data
#     split = int(split_ratio * length)
#     train_dataset = dataset.take(split).batch(batch_size)
#     val_dataset = dataset.skip(split).batch(batch_size)
#     gc.collect()
#     model.fit(train_dataset, batch_size=batch_size, epochs=number_of_epochs,use_multiprocessing=True,callbacks=[model_checkpoint],validation_data=val_dataset)


# train=mne.io.read_raw_edf(test_file)
# train=train.get_data()
# do_testowania_array=np.array(train[:19,:frame_size])

# do_testowania_array=do_testowania_array.T
# wynik = model.predict([do_testowania_array.tolist()])
# wynik=wynik[0]
# wynik=wynik.T


# for kanal in range(19):
#     t=wynik[kanal]
#     plt.plot(train[kanal,100:frame_size])
#     plt.plot(wynik[kanal,100:])
#     plt.show()