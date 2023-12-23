import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense


class ModelCreating:
    def __init__(self):
        self.train_files = ['p10_Record1.edf', 'p11_Record1.edf', 'p11_Record2.edf', 'p11_Record3.edf',
                            'p11_Record4.edf',
                            'p12_Record1.edf', 'p12_Record2.edf', 'p12_Record3.edf', 'p13_Record1.edf',
                            'p13_Record2.edf',
                            'p13_Record3.edf', 'p13_Record4.edf', 'p14_Record1.edf', 'p14_Record2.edf',
                            'p14_Record3.edf',
                            'p15_Record1.edf', 'p15_Record2.edf', 'p15_Record3.edf', 'p15_Record4.edf']
        self.test_file = 'p10_Record2.edf'
        self.frame_size = 1000
        self.split_ratio = 0.8
        self.checkpoint_filepath = "./checkpoint"
        self.number_of_epochs = 7
        self.batch_size = 32

    def build_model(self):
        model = Sequential([
            LSTM(units=70, return_sequences=True, activation="tanh", input_shape=(self.frame_size, 19)),
            LSTM(units=70, return_sequences=True, activation="tanh"),
            LSTM(units=70, return_sequences=True, activation="tanh"),
            Dense(units=19)
        ])
        model.compile(tf.keras.optimizers.Adam(), loss='mean_squared_error')
        return model

    def create_datasets(self, file, shift):
        x = tf.io.parse_tensor(tf.io.read_file(file[:-4] + '_x_shift_' + str(shift)), out_type=tf.float64)
        y = tf.io.parse_tensor(tf.io.read_file(file[:-4] + '_y_shift_' + str(shift)), out_type=tf.float64)
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = dataset.batch(self.batch_size)
        return train_dataset

    def fit_model(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        model = self.build_model()
        val_dataset = self.create_datasets(self.test_file, 0)
        for file in self.train_files:
            for i in range(0, 500, 100):
                print(file)
                train_dataset = self.create_datasets(file, i)
                model.fit(train_dataset, epochs=self.number_of_epochs, use_multiprocessing=True,
                          validation_data=val_dataset, batch_size=self.batch_size, callbacks=[tensorboard_callback]
                          )
        model.save("model.keras")


mc = ModelCreating()
mc.fit_model()
