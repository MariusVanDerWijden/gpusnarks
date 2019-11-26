cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GPUsnarks CXX CUDA)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(BEFORE ..)

cuda_add_executable(fft_test main.cpp)

if(OpenMP_CXX_FOUND)
    #set to 1 for FFT, 0 for MultiExp
    if(0)
        target_link_libraries(fft_test OpenMP::OpenMP_CXX fft_cuda fields)
    else()
        target_link_libraries(fft_test OpenMP::OpenMP_CXX multi_exp_cuda fields)
    endif()
endif()