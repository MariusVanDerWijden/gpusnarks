cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GPUsnarks CXX CUDA)

include_directories( ${MY_SOURCE_DIR}/fields)

add_library(fields SHARED field.cpp field.h)