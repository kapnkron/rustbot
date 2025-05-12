#include <torch/torch.h>
#include <iostream>

int main() {
  bool cuda_available = torch::cuda::is_available();
  std::cout << "CUDA Available: " << (cuda_available ? "true" : "false") << std::endl;

  if (cuda_available) {
    std::cout << "CUDA Device Count: " << torch::cuda::device_count() << std::endl;
    try {
      // Try to create a tensor on CUDA
      torch::Device device(torch::kCUDA);
      // Ensure the tensor is non-empty to trigger allocation and potential kernel use
      torch::Tensor tensor = torch::randn({2, 2}, device);
      std::cout << "Successfully created a tensor on CUDA: " << std::endl;
      std::cout << tensor << std::endl;
    } catch (const c10::Error& e) {
      std::cerr << "PyTorch C++ Exception during CUDA tensor operation: " << e.what() << std::endl;
      return 1;
    } catch (const std::exception& e) {
      std::cerr << "Standard C++ Exception during CUDA tensor operation: " << e.what() << std::endl;
      return 1;
    }
  }
  return 0;
} 