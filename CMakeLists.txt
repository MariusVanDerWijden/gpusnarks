cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GPUsnarks CXX CUDA)

add_subdirectory(cuda)
add_subdirectory(fields)
add_subdirectory(test)