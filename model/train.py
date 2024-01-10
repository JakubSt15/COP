import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout


class ModelCreating:
    def __init__(self):
        self.train_files = ['_1_attack.csv']
        self.test_file = '_1_attack.csv'
        self.frame_size = 1000
        self.checkpoint_filepath = "./checkpoint"
        self.number_of_epochs = 5
        self.batch_size = 100

    def build_model(self):
        model = Sequential([
            LSTM(units=19,  activation="tanh", input_shape=(self.frame_size, 19)),
            Dense(units=1, activation="tanh")
        ])
        model.compile(tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy')
        return model

    def create_datasets(self, file, shift):
        x = tf.io.parse_tensor(tf.io.read_file('./model/training_data/_1_x'), out_type=tf.float64)
        y = tf.io.parse_tensor(tf.io.read_file('./model/training_data/_1_y'), out_type=tf.int64)
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        return train_dataset

    def fit_model(self):
        model = self.build_model()
        # val_dataset = self.create_datasets(self.test_file, 0)
        for file in self.train_files:
            for i in range(0, 100, 100):
                print(file)
                train_dataset = self.create_datasets(file, i)
                model.fit(train_dataset, epochs=self.number_of_epochs, use_multiprocessing=True)
        model.save("model.keras")


mc = ModelCreating()
mc.fit_model()