import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout, Softmax, GRU
import numpy as np
import pandas as pd
from const import clr

f='model/train_data'
class ModelCreating:
    def __init__(self):
        self.train_files = ['_1_attack.csv']
        self.test_file = '_1_attack.csv'
        self.frame_size = 1024
        self.frame_step = 1024
        self.checkpoint_filepath = "./checkpoint"
        self.number_of_epochs = 1000
        self.batch_size = 100

    def build_model(self):
        model = Sequential([
            LSTM(units=15,  activation="sigmoid", return_sequences=True),
            Dropout(rate=0.2),
            LSTM(units=15,  activation="sigmoid", return_sequences=False),
            Dense(units=19, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.1))

        return model

    def create_datasets(self, verbose=0, block_on_verbose=False):
        # @todo
        all_attributes = []
        all_labels = []
        for f in [
            ['./model/cropped_records/1_training_record.csv', './model/train_data/mask_attack_1.csv'],
            ['./model/cropped_records/2_training_record.csv', './model/train_data/mask_attack_2.csv'],
            ['./model/cropped_records/4_training_record.csv', './model/train_data/mask_attack_4.csv']
            ]:
            signal_csv, labels_csv = f[0], f[1]
            attr_df = pd.read_csv(signal_csv).to_numpy().T
            labels_df = pd.read_csv(labels_csv).to_numpy().T
            attributes_frames = tf.signal.frame(attr_df, self.frame_size, self.frame_step)
            labels_frames = tf.signal.frame(labels_df, self.frame_size, self.frame_step)
            attributes_frames_reduced_mean = tf.math.reduce_mean(tf.math.square(attributes_frames), axis=2)
            labels = []
            for channel in labels_frames:
                channel_frame = np.any(channel == 1, axis=1).astype(int)
                labels.append(channel_frame.tolist())
            labels = np.array(labels)

            attributes_frames_reduced_mean_transposed, labels = attributes_frames_reduced_mean.numpy().T, labels.T
            attributes_expanded_dims = tf.expand_dims(attributes_frames_reduced_mean_transposed, axis=1, name=None)

            all_attributes.append(attributes_expanded_dims.numpy())
            all_labels.append(labels)

            if verbose > 1:
                print(clr.CYAN , "=====================Single File shapes======================")
                print(clr.CYAN ,f"| Signal File: f{f[0]}, Attack mask File: f{f[1]}" ) 
                print(clr.CYAN , "| attributes shape from df                   : ", clr.BOLD , attr_df.shape , clr.END)
                print(clr.CYAN , "|------------------------------------------------------------")
                print(clr.CYAN , "| attributes frames after tf.signal.frame    : ", clr.BOLD , attributes_frames.shape , clr.END)
                print(clr.CYAN , "| labels frames after tf.signal.frame        : ", clr.BOLD , labels_frames.shape , clr.END)
                print(clr.CYAN , "|------------------------------------------------------------")
                print(clr.CYAN , "| attributes after reduce_mean (mooc sygnału): ", clr.BOLD , attributes_frames_reduced_mean.shape , clr.END)
                print(clr.CYAN , "|------------------------------------------------------------")
                print(clr.CYAN , "| attributes frames after T                  : ", clr.BOLD , attributes_frames_reduced_mean_transposed.shape , clr.END)
                print(clr.CYAN , "| labels frames after T and minimizing       : ", clr.BOLD , labels.shape , clr.END)
                print(clr.CYAN , "|------------------------------------------------------------")
                print(clr.CYAN , "| attributes frames after expand_dims        : ", clr.BOLD , attributes_expanded_dims.shape , clr.END)
                print(clr.CYAN , "|------------------------------------------------------------")
                print(clr.END)
            

        combined_attributes = np.concatenate(all_attributes, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        if verbose > 0: 
            attacked = 0
            all = len(combined_labels)
            print(clr.GREEN,"==========================L-A-B-E-L-S========================")
            
            for l in combined_labels:
                if verbose > 2: print("| \t", l)
                if 1 in l:
                    attacked += 1  
            if verbose > 2:  print(clr.GREEN,"=============================================================")
            print(clr.GREEN,f"| All                    : {all}")
            print(clr.GREEN,f"| Frames without attack  : {all-attacked}\t| {round(100*((all-attacked)/all), 2)}%")
            print(clr.GREEN,f"| Frames with attack     : {attacked}\t| {round(100*(attacked/all), 2)}%", clr.END)

        dataset = tf.data.Dataset.from_tensor_slices((combined_attributes, combined_labels))
        dataset_shuffled = dataset.shuffle(dataset.cardinality()).batch(self.batch_size)


        if verbose > 0:
            print(clr.GREEN,"===============S-H-U-F-F-L-E-D===L-A-B-E-L-S=================")
            labels_dataset = dataset_shuffled.map(lambda x, y: y)
            attacked = 0
            all = len(combined_labels)
            
            for l in labels_dataset:
                for label in l:
                    if verbose > 2: print("| \t", label.numpy())
                    if 1 in label:
                        attacked += 1  
            if verbose > 2: print(clr.GREEN,"=============================================================")
            print(clr.GREEN,f"| All                    : {all}")
            print(clr.GREEN,f"| Frames without attack  : {all-attacked}\t| {round(100*((all-attacked)/all), 2)}%")
            print(clr.GREEN,f"| Frames with attack     : {attacked}\t| {round(100*(attacked/all), 2)}%", clr.END)

        if verbose > 0:
            print(clr.CYAN , "========================S-H-A-P-E-S==========================")
            print(clr.CYAN , "| attributes frames after concatenation      : ", clr.BOLD , combined_attributes.shape , clr.END)
            print(clr.CYAN , "| labels frames after concatenation          : ", clr.BOLD , combined_labels.shape , clr.END)
            print(clr.CYAN , "|------------------------------------------------------------")
            print(clr.CYAN , "| Train dataset after from_tensor_slices     : ", clr.BOLD , dataset , clr.END)
            print(clr.CYAN , "|------------------------------------------------------------")
            print(clr.CYAN , "| Train dataset after shuffle                : ", clr.BOLD , dataset_shuffled , clr.END)
            print(clr.CYAN , "|-----------------------------------------------------------")
            print(clr.END)
            if block_on_verbose is True: return
            
        return dataset_shuffled
       

    def create_datasets_attack_universal(self, verbose=0, block_on_verbose=False):
        # @todo
        all_attributes = []
        all_labels = []
        for f in [
            ['./model/cropped_records/1_training_record.csv', './model/train_data/mask_attack_1.csv'],
            ['./model/cropped_records/2_training_record.csv', './model/train_data/mask_attack_2.csv'],
            ['./model/cropped_records/4_training_record.csv', './model/train_data/mask_attack_4.csv']
            ]:
            signal_csv, labels_csv = f[0], f[1]
            attr_df = pd.read_csv(signal_csv).to_numpy().T
            labels_df = pd.read_csv(labels_csv).to_numpy().T
            attributes_frames = tf.signal.frame(attr_df, self.frame_size, self.frame_step)
            labels_frames = tf.signal.frame(labels_df, self.frame_size, self.frame_step)
            attributes_frames_reduced_mean = tf.math.reduce_mean(tf.math.square(attributes_frames), axis=2)
            labels = []
            for channel in labels_frames:
                print(len(channel[0]))
                return
                channel_frame = np.any(channel == 1, axis=1).astype(int)
                labels.append(channel_frame.tolist())
            labels = np.array(labels)

            attributes_frames_reduced_mean_transposed, labels = attributes_frames_reduced_mean.numpy().T, labels.T
            attributes_expanded_dims = tf.expand_dims(attributes_frames_reduced_mean_transposed, axis=1, name=None)

            all_attributes.append(attributes_expanded_dims.numpy())
            all_labels.append(labels)

            if verbose > 1:
                print(clr.CYAN , "=====================Single File shapes======================")
                print(clr.CYAN ,f"| Signal File: f{f[0]}, Attack mask File: f{f[1]}" ) 
                print(clr.CYAN , "| attributes shape from df                   : ", clr.BOLD , attr_df.shape , clr.END)
                print(clr.CYAN , "|------------------------------------------------------------")
                print(clr.CYAN , "| attributes frames after tf.signal.frame    : ", clr.BOLD , attributes_frames.shape , clr.END)
                print(clr.CYAN , "| labels frames after tf.signal.frame        : ", clr.BOLD , labels_frames.shape , clr.END)
                print(clr.CYAN , "|------------------------------------------------------------")
                print(clr.CYAN , "| attributes after reduce_mean (mooc sygnału): ", clr.BOLD , attributes_frames_reduced_mean.shape , clr.END)
                print(clr.CYAN , "|------------------------------------------------------------")
                print(clr.CYAN , "| attributes frames after T                  : ", clr.BOLD , attributes_frames_reduced_mean_transposed.shape , clr.END)
                print(clr.CYAN , "| labels frames after T and minimizing       : ", clr.BOLD , labels.shape , clr.END)
                print(clr.CYAN , "|------------------------------------------------------------")
                print(clr.CYAN , "| attributes frames after expand_dims        : ", clr.BOLD , attributes_expanded_dims.shape , clr.END)
                print(clr.CYAN , "|------------------------------------------------------------")
                print(clr.END)
            

        combined_attributes = np.concatenate(all_attributes, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        if verbose > 0: 
            attacked = 0
            all = len(combined_labels)
            print(clr.GREEN,"==========================L-A-B-E-L-S========================")
            
            for l in combined_labels:
                if verbose > 2: print("| \t", l)
                if 1 in l:
                    attacked += 1  
            if verbose > 2:  print(clr.GREEN,"=============================================================")
            print(clr.GREEN,f"| All                    : {all}")
            print(clr.GREEN,f"| Frames without attack  : {all-attacked}\t| {round(100*((all-attacked)/all), 2)}%")
            print(clr.GREEN,f"| Frames with attack     : {attacked}\t| {round(100*(attacked/all), 2)}%", clr.END)

        dataset = tf.data.Dataset.from_tensor_slices((combined_attributes, combined_labels))
        dataset_shuffled = dataset.shuffle(dataset.cardinality()).batch(self.batch_size)


        if verbose > 0:
            print(clr.GREEN,"===============S-H-U-F-F-L-E-D===L-A-B-E-L-S=================")
            labels_dataset = dataset_shuffled.map(lambda x, y: y)
            attacked = 0
            all = len(combined_labels)
            
            for l in labels_dataset:
                for label in l:
                    if verbose > 2: print("| \t", label.numpy())
                    if 1 in label:
                        attacked += 1  
            if verbose > 2: print(clr.GREEN,"=============================================================")
            print(clr.GREEN,f"| All                    : {all}")
            print(clr.GREEN,f"| Frames without attack  : {all-attacked}\t| {round(100*((all-attacked)/all), 2)}%")
            print(clr.GREEN,f"| Frames with attack     : {attacked}\t| {round(100*(attacked/all), 2)}%", clr.END)

        if verbose > 0:
            print(clr.CYAN , "========================S-H-A-P-E-S==========================")
            print(clr.CYAN , "| attributes frames after concatenation      : ", clr.BOLD , combined_attributes.shape , clr.END)
            print(clr.CYAN , "| labels frames after concatenation          : ", clr.BOLD , combined_labels.shape , clr.END)
            print(clr.CYAN , "|------------------------------------------------------------")
            print(clr.CYAN , "| Train dataset after from_tensor_slices     : ", clr.BOLD , dataset , clr.END)
            print(clr.CYAN , "|------------------------------------------------------------")
            print(clr.CYAN , "| Train dataset after shuffle                : ", clr.BOLD , dataset_shuffled , clr.END)
            print(clr.CYAN , "|-----------------------------------------------------------")
            print(clr.END)
            if block_on_verbose is True: return
            
        return dataset_shuffled
    
    def fit_model(self, verbose=0, block_on_verbose=False):
        model = self.build_model()
        # train_dataset = self.create_datasets(verbose, block_on_verbose)
        train_dataset = self.create_datasets_attack_universal(verbose, block_on_verbose)
        if not train_dataset: return
        model.fit(train_dataset, epochs=self.number_of_epochs, use_multiprocessing=True)
        model.save("model.keras")


mc = ModelCreating()
mc.fit_model(verbose=3, block_on_verbose=True)
        
# import matplotlib.pyplot as plt
# attr_df = pd.read_csv('./model/cropped_records/2_training_record.csv').to_numpy().T

# plt.plot(attr_df[4])
# plt.plot(attr_df[7])
# plt.plot(attr_df[11])
# plt.plot(attr_df[15])
# plt.show()

