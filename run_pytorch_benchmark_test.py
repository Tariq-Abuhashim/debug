import time
import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# Timing the operation on the CPU
x_cpu = torch.randn(10000, 10000)
start_time = time.time()
y_cpu = torch.matmul(x_cpu, x_cpu)
print("CPU computation time:", time.time() - start_time)

# Timing the operation on the GPU
if cuda_available:
    x_gpu = x_cpu.to('cuda')
    start_time = time.time()
    y_gpu = torch.matmul(x_gpu, x_gpu)
    torch.cuda.synchronize()  # Wait for GPU to finish computation
    print("GPU computation time:", time.time() - start_time)

