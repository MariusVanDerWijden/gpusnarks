set(CMAKE_C_COMPILER /usr/bin/gcc-6)
set(CMAKE_CXX_COMPILER /usr/bin/g++-6)

find_package(CUDA REQUIRED)


cuda_add_executable(fft_cuda fft_kernel.cu)
