import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import sklearn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("All imports successful!")

# Create a simple neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
# Create a model instance
model = SimpleNN()
print("Model created successfully:")
print(model)

# Generate some random data
X = np.random.randn(100, 10)
y = np.random.randn(100, 1)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Create a loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train for a few iterations
for epoch in tqdm(range(10), desc="Training"):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

print("Training completed successfully!")

# Create a simple plot
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y)), y, color='blue', label='True')
plt.scatter(range(len(y)), model(X_tensor).detach().numpy(), color='red', label='Predicted')
plt.legend()
plt.title("Simple Regression Visualization")
plt.savefig("test_plot.png")
plt.close()

print("Plot created and saved successfully!")
print("\nAll components verified! Your Neural Architecture Search environment is ready.") 