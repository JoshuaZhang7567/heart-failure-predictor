import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv

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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Assuming read_data() function is defined as in your original code

data_input, data_output = read_data()

# Convert data to PyTorch tensors
X = torch.FloatTensor(data_input)
y = torch.FloatTensor(data_output)

# Initialize the model
torch.manual_seed(69)
model = Model()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2500
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training completed!")

# Test the model
model.eval()
with torch.no_grad():
    for i in range(0,10):
        test_input = torch.FloatTensor([data_input[i]])  # Example input
        prediction = model(test_input)
        print(f"Prediction for test input: {prediction.numpy()}")
