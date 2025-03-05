import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

if cuda_available:
    # Create a tensor and move it to the GPU
    x = torch.tensor([1.0, 2.0, 3.0])
    x = x.to('cuda')
    print("Tensor on GPU:", x)

    # Perform a simple operation on the GPU
    y = x ** 2
    print("Tensor operation result on GPU:", y)

    # Move tensor back to CPU
    y = y.to('cpu')
    print("Tensor moved back to CPU:", y)
else:
    print("CUDA is not available. Skipping GPU tests.")
