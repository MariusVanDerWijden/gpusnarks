
find_package(CUDA REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(CMAKE_C_COMPILER /usr/bin/gcc-6)
set(CMAKE_CXX_COMPILER /usr/bin/g++-6)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=sm_50")

include_directories( ${MY_SOURCE_DIR})

cuda_add_executable(fft_cuda fft_kernel.cu)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fft_cuda PUBLIC OpenMP::OpenMP_CXX)
endif()



