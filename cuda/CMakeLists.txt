cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GPUsnarks CXX CUDA)
enable_language(CUDA)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v; -lineinfo; -use_fast_math)
list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)

include_directories(${MY_SOURCE_DIR}/cuda)

cuda_add_library(fft_cuda STATIC fft_kernel.cu fft_kernel.h device_field.h)
target_include_directories(fft_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

cuda_add_library(multi_exp_cuda STATIC multi_exp.cu multi_exp.h device_field.h)
target_include_directories(multi_exp_cuda PUBLIC ${CUDA_INCLUDE_DIRS})
