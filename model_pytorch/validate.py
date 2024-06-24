from prepare_data import prepare_dataset_attack_model
from sklearn.metrics import accuracy_score
import torch
from train_attack import AttackModel, MultiChannelAttackModel  # Import the AttackModel class from the module
import matplotlib.pyplot as plt
from prepare_data import get_attack_sample_from_predictions
data =  [
    ['./model_pytorch/records/1_training_record.csv', './model_pytorch/labels/1_training_labels.csv'],
]

validation_attributes, validation_labels = prepare_dataset_attack_model(data, plot_verbose=False)

model_save_path = './model_pytorch/attack_model_pyTorch.pth'

loaded_model = AttackModel()
loaded_model.load_state_dict(torch.load(model_save_path))

validation_attributes = torch.tensor(validation_attributes, dtype=torch.float32)
validation_labels = torch.tensor(validation_labels, dtype=torch.float32)
validation_logits = None

print(validation_attributes.shape)
vector = torch.linspace(-1, 1, 560)
validation_attributes = vector.unsqueeze(1).repeat(1, 19)
print(validation_attributes.shape)
input()
with torch.no_grad():
    loaded_model.eval()
    validation_logits = loaded_model(validation_attributes).squeeze()
    validation_predictions = torch.round(validation_logits)

validation_accuracy = accuracy_score(validation_labels.numpy(), validation_predictions.numpy())
import numpy as np

channel = 10
for j in range(7,8):
    fig, axs = plt.subplots(2,1)
    axs[0].plot(np.array(validation_attributes).T[j])
    axs[0].set_title('Atak wskazany przez doktora')
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


print(validation_predictions)






