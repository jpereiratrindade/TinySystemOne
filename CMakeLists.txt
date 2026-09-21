cmake_minimum_required(VERSION 3.25)
project(TinySystemOne
    VERSION 0.1.0
    DESCRIPTION "A white-box C++26 engine to study calibrated probabilistic judgments and uncertainty"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization and Warning flags
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -O3
    )
endif()

# Include directories
include_directories(include)

# Main Experiment Runner (EXP-001)
add_executable(tso_exp001 src/main.cpp)
target_include_directories(tso_exp001 PRIVATE include)

# Unit Tests Targets
add_executable(test_math tests/test_math.cpp)
target_include_directories(test_math PRIVATE include)

add_executable(test_dataset tests/test_dataset.cpp)
target_include_directories(test_dataset PRIVATE include)

enable_testing()
add_test(NAME TestMath COMMAND test_math)
add_test(NAME TestDataset COMMAND test_dataset)
add_test(NAME Exp001 COMMAND tso_exp001)
