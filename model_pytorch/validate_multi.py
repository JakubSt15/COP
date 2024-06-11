from prepare_data import prepare_datasets_channel_attacks
from sklearn.metrics import accuracy_score
import torch
from train_attack import MultiChannelAttackModel  # Import the AttackModel class from the module
import matplotlib.pyplot as plt
data =  [
    ['./model_pytorch/records/1_training_record.csv', './model_pytorch/labels_multi/mask_attack_1.csv'],
]

validation_data = prepare_datasets_channel_attacks(data, plot_verbose=False)
channel = 5
validation_attributes = validation_data[channel]["attr"]
validation_labels = validation_data[channel]["labels"]

model_save_path = './model_pytorch/multi_channel_model_pyTorch.pth'

loaded_model = MultiChannelAttackModel()
loaded_model.load_state_dict(torch.load(model_save_path))

validation_attributes = torch.tensor(validation_attributes, dtype=torch.float32)
validation_labels = torch.tensor(validation_labels, dtype=torch.float32)
validation_logits = None

with torch.no_grad():
    loaded_model.eval()
    validation_logits = loaded_model(validation_attributes).squeeze()
    print(validation_logits)
    validation_predictions = torch.round(validation_logits)
    
validation_accuracy = accuracy_score(validation_labels.numpy(), validation_predictions.numpy())
import numpy as np

channel = 1
for j in range(7,8):
    fig, axs = plt.subplots(2,1)
    axs[0].plot(np.array(validation_attributes).T[j])
    axs[0].set_title('Atak określony przez K-means')
    # Iterate through validation labels and change color based on the label
    for i in range(len(validation_labels)):
        if validation_labels[i] == 1:
            axs[0].plot(i, validation_attributes[i][j], 'ro')  # Red color for label 1
        else:
            axs[0].plot(i, validation_attributes[i][j], 'bo')  # Blue color for label 0


    axs[1].plot(np.array(validation_attributes).T[j])
    axs[1].set_title('Atak przewidziany przez model')
    # Iterate through validation labels and change color based on the label
    print(len(validation_predictions))
    for i in range(len(validation_predictions)):
        if validation_predictions[i] > 0.5:
            axs[1].plot(i, validation_attributes[i][j], 'ro')  # Red color for label 1
        else:
            axs[1].plot(i, validation_attributes[i][j], 'bo')  # Blue color for label 0

    plt.show()


print("Validation Accuracy:", validation_accuracy)

#Todo:
# okreslic start-sample i end-sample przewidzianego ataku i wycicac tylko ten fragment sygnalu dla jakzdego kanalu