cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GPUsnarks LANGUAGES CXX CUDA)

find_package(OpenMP)
find_package(CUDA REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

cuda_add_library(fft_cuda STATIC
    ${PROJECT_SOURCE_DIR}/cuda/fft_kernel.cu
    ${PROJECT_SOURCE_DIR}/cuda/fft_kernel.h)

target_compile_features(fft_cuda PUBLIC cxx_std_14)
set_target_properties( fft_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(fft_cuda PUBLIC ${CUDA_INCLUDE_DIRS})

add_executable(fft_test main.cpp)

set_property(TARGET fft_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(fft_test PRIVATE fft_cuda)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fft_test PUBLIC OpenMP::OpenMP_CXX )
endif()



