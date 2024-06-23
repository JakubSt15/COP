from prepare_data import prepare_dataset_attack_model, prepare_datasets_channel_attacks
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.model = nn.Sequential( 
            nn.Linear(19, 32),
            nn.Dropout(0.01),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class AttackTrainer:
    def __init__(self, data, epochs=10000, learning_rate=0.6, dropout=0.18, test_size=0.2, model_save_path='./model_pytorch/attack_model_pyTorch.pth'):
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

    def fit_attack(self, validation_split=0.2, plot_loss=False):
        # Splitting data into train and validation sets
        
        attributes, labels = prepare_dataset_attack_model(self.data, shuffle=False, plot_verbose=False)

        pd.DataFrame(attributes).to_csv('ATRYBUTY.csv', index=False, header=False)
        
        split_index = int(len(attributes) * (1 - validation_split))
        train_attributes, val_attributes = attributes[:split_index], attributes[split_index:]
        train_labels, val_labels = labels[:split_index], labels[split_index:]

        train_attributes = torch.tensor(train_attributes, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        val_attributes = torch.tensor(val_attributes, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.float32)

        model = AttackModel()
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=self.learning_rate)
        
        trainingEpoch_loss, validationEpoch_loss = [], []

        for epoch in range(self.epochs):
            model.train()
            train_logits = model(train_attributes).squeeze()
            train_loss = loss_fn(train_logits, train_labels)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            trainingEpoch_loss.append(train_loss.detach().numpy())
            # Validation
            model.eval()
            val_logits = model(val_attributes).squeeze()
            val_loss = loss_fn(val_logits, val_labels)
            validationEpoch_loss.append(val_loss.detach().numpy())
            
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | Validation Loss: {val_loss:.5f}")

        torch.save(model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

        if plot_loss: 
            plt.plot(trainingEpoch_loss, label='train_loss')
            plt.plot(validationEpoch_loss, label='val_loss')
            plt.legend()
            plt.show()

        return model

class MultiChannelAttackModel(nn.Module):
    def __init__(self):
        super(MultiChannelAttackModel, self).__init__()
        self.model = nn.Sequential( 
            nn.Linear(18, 50),
            nn.Dropout(0.01),
            nn.Sigmoid(),
            nn.Linear(50, 120),
            nn.Sigmoid(),
            nn.Linear(120, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
class MultiChannelAttackTrainer:
    def __init__(self, attr, labels, epochs=35000, learning_rate=0.5, dropout=0.18, test_size=0.2, model_save_path='./model_pytorch/multi_channel_model_pyTorch.pth'):
        self.attr = attr
        self.labels = labels
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.test_size = test_size
        self.model_save_path = model_save_path

    def accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() 
        acc = (correct / len(y_pred)) * 100 
        return acc

    def fit_attack(self, validation_split=0.2, plot_loss=False):
        # Splitting data into train and validation sets
        attributes = self.attr
        labels = self.labels

        split_index = int(len(attributes) * (1 - validation_split))
        train_attributes, val_attributes = attributes[:split_index], attributes[split_index:]
        train_labels, val_labels = labels[:split_index], labels[split_index:]

        train_attributes = torch.tensor(train_attributes, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        val_attributes = torch.tensor(val_attributes, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.float32)

        print(f"LENGTHS: {len(val_attributes)}, {len(val_labels)}")
        model = MultiChannelAttackModel()
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=self.learning_rate)
        
        trainingEpoch_loss, validationEpoch_loss = [], []

        for epoch in range(self.epochs):
            model.train()
            train_logits = model(train_attributes).squeeze()
            train_loss = loss_fn(train_logits, train_labels)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            trainingEpoch_loss.append(train_loss.detach().numpy())
            # Validation
            model.eval()
            val_logits = model(val_attributes).squeeze()
            val_loss = loss_fn(val_logits, val_labels)
            validationEpoch_loss.append(val_loss.detach().numpy())
            
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | Validation Loss: {val_loss:.5f}")

        torch.save(model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

        if plot_loss: 
            plt.plot(trainingEpoch_loss, label='train_loss')
            plt.plot(validationEpoch_loss, label='val_loss')
            plt.legend()
            plt.show()

        return model


# Example usage:
data =  [
    ['./model_pytorch/records/1_training_record.csv', './model_pytorch/labels/1_training_labels.csv'],
    ['./model_pytorch/records/2_training_record.csv', './model_pytorch/labels/2_training_labels.csv'],
    ['./model_pytorch/records/4_training_record.csv', './model_pytorch/labels/4_training_labels.csv'],
    ['./model_pytorch/records/5_training_record.csv', './model_pytorch/labels/5_training_labels.csv'],
    ['./model_pytorch/records/6_training_record.csv', './model_pytorch/labels/6_training_labels.csv'],
    ['./model_pytorch/records/7_training_record.csv', './model_pytorch/labels/7_training_labels.csv'],
    ['./model_pytorch/records/44_training_record.csv', './model_pytorch/labels/44_training_labels.csv']
]

data_multi =  [
    ['./model_pytorch/records/1_training_record.csv', './model_pytorch/labels_multi/mask_attack_1.csv'],
    ['./model_pytorch/records/2_training_record.csv', './model_pytorch/labels_multi/mask_attack_2.csv'],
    # ['./model_pytorch/records/4_training_record.csv', './model_pytorch/labels_multi/mask_attack_4.csv']
]
trainer = AttackTrainer(data)
model = trainer.fit_attack(plot_loss=True)  

# attributes, labels = prepare_dataset_attack_model(data, shuffle=False, plot_verbose=True)


# x = prepare_datasets_channel_attacks(data_multi, plot_verbose=False, fast=False)

# channel = 5
# attr = x[channel]["attr"]
# labels = x[channel]["labels"]

# trainer = MultiChannelAttackTrainer(attr, labels)
# model = trainer.fit_attack(plot_loss=True)  
