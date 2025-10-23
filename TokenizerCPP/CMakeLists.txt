cmake_minimum_required(VERSION 3.31)
project(TokenizerCPP)

set(CMAKE_CXX_STANDARD 17)

add_executable(TokenizerCPP main.cpp
        Tokenizer.cpp
        Tokenizer.h)
