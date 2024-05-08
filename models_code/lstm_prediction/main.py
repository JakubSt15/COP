import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from keras import  Model
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.python.keras.regularizers import l1_l2,l2
from datarefiner import DataRefiner
from datarefiner import DataRefiner
import tensorflow as tf
FREQUENCY=512
FRAME_SIZE=FREQUENCY*1
BATCH_SIZE=5
EPOCHS=1000

#%%

class PreparingDatasets:
    def __init__(self):
        self.frequency = FREQUENCY
        self.frame_size = FRAME_SIZE
        self.dataRefiner = DataRefiner()

    def generate_dataset(self, directory):
        data_x = []
        data_y=[]
        training_samples=32
        for file in os.listdir(directory):
            full_path = os.path.join(directory, file)
            attack = pd.read_csv(full_path)
            attack = attack.to_numpy().T.astype('float32')
            attack = self.dataRefiner.refine(attack).T

            attack_length = len(attack)
            remainder = attack_length % training_samples
            attack=attack.T
            if remainder != 0:
                attack = attack[:,:-remainder]
            frames = tf.signal.frame(attack, frame_length=training_samples, frame_step=training_samples, pad_end=False)
            frames = tf.transpose(frames, perm=[1, 2, 0])
            frames=np.array(frames)
            if frames.shape[0] <= 1:
                print(f"Skipping file {file} as it resulted in only one frame.")
                continue
            # first_channel = frames[:, :, 0]

            # # Flatten the data for plotting
            # first_channel_flattened = first_channel.flatten()

            # # Create the plot
            # plt.figure(figsize=(10, 6))
            # plt.plot(first_channel_flattened)
            # plt.title(f'First Channel Data (Array shape: {frames.shape})')
            # plt.xlabel('Time')
            # plt.ylabel('Amplitude')
            # plt.show()

            data_x.append(np.array(frames[:-1]))
            data_y.append(np.array(frames[1:]))
        
        
        data_x= np.concatenate(data_x, axis=0)
        data_y= np.concatenate(data_y, axis=0)

        dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))
        dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)
        return dataset


class LSTMModel:
    def __init__(self):
        self.number_of_epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.frame_size = FRAME_SIZE

    def build_model(self):
        model = Sequential([
            LSTM(units=64, return_sequences=True),
            BatchNormalization(),
            LSTM(units=32, return_sequences=True),
            BatchNormalization(),
            Dense(units=19, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.AdamW(), loss='mse', metrics=['mae'])
        return model

    def fit_model(self, train_dataset, validation_dataset):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        train_dataset = train_dataset.batch(self.batch_size)
        validation_dataset = validation_dataset.batch(self.batch_size)

        model = self.build_model()
        model.fit(train_dataset, validation_data=train_dataset, epochs=self.number_of_epochs,
                  callbacks=[tensorboard_callback,early_stopping])
        model.save("model.keras")
        return model


pr = PreparingDatasets()
train_dataset = pr.generate_dataset("./train")
val_dataset = pr.generate_dataset("./validation")
test_dataset = pr.generate_dataset("./test")

model = LSTMModel()
model = model.fit_model(train_dataset, val_dataset)
x_test, y_test = tuple(zip(*train_dataset))
x_test = np.array(x_test)
y_test = np.array(y_test)
model.evaluate(x_test,y_test)
y_pred=model.predict(x_test)
print(y_pred.shape)
y_pred = y_pred.reshape(-1, 19)
y_test = y_test.reshape(-1, 19)
plt.plot(y_pred[:1000,0])
plt.plot(y_test[:1000,0])
plt.show()