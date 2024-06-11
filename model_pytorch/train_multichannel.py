from prepare_data import prepare_dataset_attack_model, prepare_datasets_channel_attacks
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np

class SingleChannelModel(nn.Module):
    def __init__(self):
        super(SingleChannelModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=19, out_features=19*2),
            nn.ReLU(),
            nn.Linear(in_features=19*2, out_features=16),
            nn.Dropout(0.18),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
        )

    def forward(self, x):
        return self.model(x)

class SingleChannelTrainer:
    def __init__(self, data, epochs=4000, learning_rate=0.015, dropout=0.18, test_size=0.2, model_save_path='./model_pytorch/model_pyTorch.pth'):
        self.data = data
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.test_size = test_size
        self.model_save_path = model_save_path

    def accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() 
        acc = (correct / len(y_pred)) * 100 
        return acc

    def fit_attack(self, CHANNEL=0):
        attributes, labels, channel_names = prepare_datasets_channel_attacks(self.data, shuffle=True)




        train_attributes, test_attributes, train_labels, test_labels = train_test_split(attributes, labels, test_size=self.test_size)
        train_attributes = torch.tensor(train_attributes, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        test_attributes = torch.tensor(test_attributes, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)
        model = SingleChannelModel()
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            model.train()
            y_logits = model(train_attributes).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))
            loss = loss_fn(y_logits, train_labels) 
            acc = self.accuracy_fn(y_true=train_labels, y_pred=y_pred) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            if test_attributes is not None and test_labels is not None:
                with torch.no_grad():
                    test_pred = model(test_attributes).squeeze() 
                    test_loss = loss_fn(test_pred, test_labels)
                if epoch % 100 == 0:
                    print(f"Epoch: {epoch} | Loss: {loss:.5f}| Test loss: {test_loss:.5f}")
            else:
                if epoch % 100 == 0:
                    print(f"Epoch: {epoch} | Loss: {loss:.5f}")

        torch.save(model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        return model

# Example usage:
data =  [
    ['./model/cropped_records/1_training_record.csv', './model/train_data/mask_attack_1.csv'],
    ['./model/cropped_records/2_training_record.csv', './model/train_data/mask_attack_2.csv'],
    ['./model/cropped_records/4_training_record.csv', './model/train_data/mask_attack_4.csv']
]
x = prepare_datasets_channel_attacks(data)
for i in range(len(x)):
    for j in range(len(x[i])):
        print(x[i][j]['channelName'])
# trainer = SingleChannelTrainer(data)
# model = trainer.fit_attack()