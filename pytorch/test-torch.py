import torch
import torch.nn as nn
import torch.optim as optim

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define a simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, output_size=2):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate model
model = SimpleMLP().to(device)

# Generate random input and target
batch_size = 64
input_data = torch.randn(batch_size, 10).to(device)
target = torch.randint(0, 2, (batch_size,)).to(device)  # binary classification

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop (5 epochs)
print("Training...")
for epoch in range(5):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Inference
print("\nInference on new random data:")
test_input = torch.randn(5, 10).to(device)
with torch.no_grad():
    predictions = model(test_input)
    predicted_classes = torch.argmax(predictions, dim=1)
print("Predicted classes:", predicted_classes.cpu().numpy())
