import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from sklearn.model_selection import train_test_split


def format_sex(string):
    if string == 'M':
        return 1
    return 0

def format_chestpaintype(string):
    if string == 'TA':
        return 0
    elif string == 'ATA':
        return 1
    elif string == 'NAP':
        return 2
    return 3

def format_ECG(string):
    if string == 'Normal':
        return 0
    elif string == 'ST':
        return 1
    return 2

def format_exerciseangina(string):
    if string == 'Y':
        return 1
    return 0

def format_slope(string):
    if string == 'Up':
        return 0
    elif string == 'Flat':
        return 1
    return 2

def read_data():
    data_input = []
    data_output = []
    with open("heart.csv") as file_in:
        reader = csv.reader(file_in)
        for line in reader:
            line[0] = int(line[0])
            line[1] = format_sex(line[1])
            line[2] = format_chestpaintype(line[2])
            line[3] = int(line[3])
            line[4] = int(line[4])
            line[5] = int(line[5])
            line[6] = format_ECG(line[6])
            line[7] = int(line[7])
            line[8] = format_exerciseangina(line[8])
            line[9] = float(line[9])
            line[10] = format_slope(line[10])
            line[11] = int(line[11])
            data_input.append(line[:11])
            data_output.append(line[11])
    
    return data_input, data_output

class Model(nn.Module):
    def __init__(self, in_features=11, h1=11, h2=11, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))  # Add sigmoid activation
        return x

data_input, data_output = read_data()

X = torch.FloatTensor(data_input)
y = torch.FloatTensor(data_output).view(-1, 1)  # Reshape to [n_samples, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

torch.manual_seed(69)
model = Model()

criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.AdamW(model.parameters(), lr=0.0035)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

num_epochs = 5000
batch_size = 64

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i in range(0, len(X), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        correct_predictions += (predicted == batch_y).sum().item()
        total_predictions += batch_y.size(0)
    
    epoch_loss = total_loss / (len(X) // batch_size)
    epoch_accuracy = correct_predictions / total_predictions

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    scheduler.step()

print("Training completed!")

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(len(X_test)):
        test_input = X_test[i].unsqueeze(0)  # Add batch dimension
        prediction = model(test_input)
        predicted_class = 1 if prediction.item() > 0.5 else 0
        actual_class = y_test[i].item()
        
        if predicted_class == actual_class:
            correct += 1
        total += 1
        
        if i < 10:  # Print first 10 predictions
            print(f"Input: {test_input.numpy()}")
            print(f"Prediction: {prediction.item():.4f}, Class: {predicted_class}, Actual: {actual_class}")
            print()

    print(f"Test Accuracy: {correct/total:.4f}")
