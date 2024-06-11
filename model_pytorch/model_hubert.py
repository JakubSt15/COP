import torch
import numpy as np
import pandas as pd
from train_attack import AttackModel,MultiChannelAttackModel
import tensorflow as tf
model_predykcja = tf.keras.models.load_model('model.keras')

attack = pd.read_csv('./model_pytorch/records/40_training_record.csv')
print(attack)
a=np.array(attack)
a=a[np.newaxis,:4]
print(a.shape)
y=model_predykcja.predict(a)


loaded_model1 = AttackModel()
loaded_model1.load_state_dict(torch.load( './model_pytorch/attack_model_pyTorch.pth'))
tensor1=torch.tensor(y,dtype=torch.float32)
print(tensor1)
with torch.no_grad():
    loaded_model1.eval()
    validation_logits = loaded_model1(tensor1).squeeze()
    validation_predictions = torch.round(validation_logits)

loaded_model2 = MultiChannelAttackModel()
loaded_model2.load_state_dict(torch.load('./model_pytorch/multi_channel_model_pyTorch.pth'))

with torch.no_grad():
    loaded_model2.eval()
    validation_logits = loaded_model2(validation_predictions).squeeze()
    print(validation_logits)
    validation_predictions = torch.round(validation_logits)