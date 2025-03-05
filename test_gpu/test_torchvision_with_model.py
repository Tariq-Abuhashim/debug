import torch
import torchvision.models as models

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Move the model to GPU
if cuda_available:
    model = model.to('cuda')

# Create a random input tensor
input_tensor = torch.randn(1, 3, 224, 224)

# Move the input tensor to GPU
if cuda_available:
    input_tensor = input_tensor.to('cuda')

# Perform a forward pass with the model
output = model(input_tensor)
print("Output shape:", output.shape)
