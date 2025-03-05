import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# Check the CUDA version
cuda_version = torch.version.cuda
print("CUDA Version:", cuda_version)

# Check the number of available GPUs
gpu_count = torch.cuda.device_count()
print("Number of GPUs available:", gpu_count)

# Check the name of the GPU(s)
if cuda_available:
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

