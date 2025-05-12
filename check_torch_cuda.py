import torch

print(f"--- Python PyTorch CUDA Check ---")
print(f"PyTorch version: {torch.__version__}")

try:
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")

    if cuda_available:
        print(f"CUDA version PyTorch built with: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        current_dev_idx = torch.cuda.current_device()
        print(f"Current CUDA device index: {current_dev_idx}")
        dev_name = torch.cuda.get_device_name(current_dev_idx)
        print(f"Current CUDA device name: {dev_name}")

        print("\nAttempting to create a tensor on CUDA...")
        # Basic tensor creation
        x = torch.randn(3, 3, device='cuda')
        print("Successfully created tensor x on CUDA:")
        print(x)
        print(f"x.is_cuda: {x.is_cuda}, x.device: {x.device}")

        # Another way to move to CUDA
        y = torch.tensor([1.0, 2.0, 3.0])
        print(f"\nCreated tensor y on CPU: {y}, device: {y.device}")
        y = y.cuda()
        print("Moved tensor y to CUDA:")
        print(y)
        print(f"y.is_cuda: {y.is_cuda}, y.device: {y.device}")

        # Simple operation
        z = x + y
        print("\nPerformed x + y on CUDA:")
        print(z)
        print(f"z.is_cuda: {z.is_cuda}, z.device: {z.device}")

        print("\n--- CUDA Check Successful ---")

    else:
        print("\nCUDA is NOT available to PyTorch in this Python environment.")
        # Try to get more diagnostic info if CUDA is expected
        try:
            torch.randn(1).cuda()
        except Exception as e:
            print(f"Error when trying to force CUDA operation: {e}")


except Exception as e:
    print(f"\nAN ERROR OCCURRED DURING PYTORCH CUDA CHECK: {e}")
    import traceback
    traceback.print_exc()

print("--- End of Check ---") 