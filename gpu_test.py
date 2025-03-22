import torch
import time

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test CPU performance
print("\nTesting CPU performance...")
start_time = time.time()
x_cpu = torch.randn(1000, 1000)
y_cpu = torch.randn(1000, 1000)
for _ in range(20):
    z_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.time() - start_time
print(f"CPU computation time: {cpu_time:.4f} seconds")

# Test GPU if available
if torch.cuda.is_available():
    print("\nTesting GPU performance...")
    print("CUDA version:", torch.version.cuda)
    print("GPU device:", torch.cuda.get_device_name(0))
    
    # Move tensors to GPU
    x_gpu = torch.randn(1000, 1000, device="cuda")
    y_gpu = torch.randn(1000, 1000, device="cuda")
    
    # Warm-up
    for _ in range(5):
        z_gpu = torch.matmul(x_gpu, y_gpu)
    
    # Benchmark
    start_time = time.time()
    for _ in range(20):
        z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start_time
    
    print(f"GPU computation time: {gpu_time:.4f} seconds")
    print(f"GPU is {cpu_time/gpu_time:.1f}x faster than CPU")
else:
    print("\nNo GPU available. Install PyTorch with CUDA support to use your RTX 3060.") 