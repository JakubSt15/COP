from prepare_data import prepare_dataset_attack_model
from sklearn.metrics import accuracy_score
import torch
from train_attack import AttackModel, MultiChannelAttackModel  # Import the AttackModel class from the module
import matplotlib.pyplot as plt
from prepare_data import get_attack_sample_from_predictions, prepare_prediction_multi_channel_datasets
import pandas as pd
import numpy as np
import tensorflow as tf
from visualize import visualize_predicted_attack

data =  [
    ['./model_pytorch/records/1_training_record.csv', './model_pytorch/labels/1_training_labels.csv'],
]


model_predykcja = tf.keras.models.load_model('model.keras')

attack = pd.read_csv(data[0][0])
attack, _ = prepare_dataset_attack_model(data, plot_verbose=False)
a=np.array(attack)
a=a[np.newaxis,:4]
print(a.shape)
print(a)
input()
y=model_predykcja.predict(a)



# validation_attributes, _ = prepare_dataset_attack_model(data, plot_verbose=False)

attac_model_save_path = './model_pytorch/attack_model_pyTorch.pth'

loaded_model = AttackModel()
loaded_model.load_state_dict(torch.load(attac_model_save_path))

validation_attributes = torch.tensor(y, dtype=torch.float32)
validation_logits = None

with torch.no_grad():
    loaded_model.eval()
    validation_logits = loaded_model(validation_attributes).squeeze()
    validation_predictions = torch.round(validation_logits)

start_sample, end_sample = get_attack_sample_from_predictions(validation_predictions, FRAME_SIZE=1000)
attr_df = pd.read_csv(data[0][0], header=None)[start_sample:end_sample]

visualize_predicted_attack(y[0], validation_predictions)


'''
    Multi channel
'''

validation_data = prepare_prediction_multi_channel_datasets(attr_df, plot_verbose=False, rollingN=10)
channel = 5



final_predictions = []
for i in range(len(validation_data)):
    validation_attributes = validation_data[i]["attr"]

    model_save_path = './model_pytorch/multi_channel_model_pyTorch.pth'

    loaded_model = MultiChannelAttackModel()
    loaded_model.load_state_dict(torch.load(model_save_path))

    validation_attributes = torch.tensor(validation_attributes, dtype=torch.float32)
    validation_logits = None

    with torch.no_grad():
        loaded_model.eval()
        validation_logits = loaded_model(validation_attributes).squeeze()
        validation_predictions = torch.round(validation_logits)

    final_predictions.append(validation_predictions.numpy())

final_predictions_df = pd.DataFrame(np.array(final_predictions))
print(final_predictions_df)