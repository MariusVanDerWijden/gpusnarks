find_package(CUDA REQUIRED)

cuda_add_executable(fft_cuda fft_kernel.cu)