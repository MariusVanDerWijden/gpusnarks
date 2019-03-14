cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
#project(GPUsnarks CXX CUDA)
enable_language(CUDA)


find_package(CUDA REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

cuda_add_library(fft_cuda SHARED fft_kernel.cu fft_kernel.h)
target_link_libraries(fft_cuda ${CUDA_LIBRARIES})

cuda_add_executable(fft_test main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fft_test OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})
endif()