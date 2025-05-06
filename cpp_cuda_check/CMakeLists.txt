cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cuda_check)

# Point CMake to the directory containing TorchConfig.cmake
set(CMAKE_PREFIX_PATH "/home/morgan/libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

add_executable(cuda_check cuda_check.cpp)
target_link_libraries(cuda_check "${TORCH_LIBRARIES}")
set_property(TARGET cuda_check PROPERTY CXX_STANDARD 17) 