import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor x:", x)

# Move tensor to GPU if available
if torch.cuda.is_available():
    x = x.to("cuda")
    print("Tensor x on GPU:", x)
else:
    print("Running on CPU only.")

# Simple operation
y = x * 2
print("Tensor y (x * 2):", y)
