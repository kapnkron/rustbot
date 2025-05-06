#include <torch/torch.h>
#include <iostream>

int main() {
  bool cuda_available = torch::cuda::is_available();
  std::cout << "CUDA available (LibTorch C++ API): " << (cuda_available ? "true" : "false") << std::endl;

  if (cuda_available) {
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
  }

  return 0;
} 