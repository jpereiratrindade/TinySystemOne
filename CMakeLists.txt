cmake_minimum_required(VERSION 3.25)
project(TinySystemOne
    VERSION 0.4.0
    DESCRIPTION "A white-box C++26 engine to study calibrated probabilistic judgments and uncertainty"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_SANITIZERS "Enable Address and Undefined Behavior sanitizers in Debug" OFF)

# Compiler Warning Flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
    )

    if(ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE MATCHES "Debug")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# Include directories
include_directories(include)

# Main Experiment Runners
add_executable(tso_exp001 src/main.cpp)
target_include_directories(tso_exp001 PRIVATE include)

add_executable(tso_exp002 src/exp002_main.cpp)
target_include_directories(tso_exp002 PRIVATE include)

add_executable(tso_exp003 src/exp003_main.cpp)
target_include_directories(tso_exp003 PRIVATE include)

add_executable(tso_exp004 src/exp004_main.cpp)
target_include_directories(tso_exp004 PRIVATE include)

# Unit & Acceptance Tests Targets
add_executable(test_math tests/test_math.cpp)
target_include_directories(test_math PRIVATE include)

add_executable(test_dataset tests/test_dataset.cpp)
target_include_directories(test_dataset PRIVATE include)

add_executable(test_acceptance tests/test_acceptance.cpp)
target_include_directories(test_acceptance PRIVATE include)

enable_testing()
add_test(NAME TestMath COMMAND test_math)
add_test(NAME TestDataset COMMAND test_dataset)
add_test(NAME TestAcceptance COMMAND test_acceptance)
add_test(NAME Exp001 COMMAND tso_exp001)
add_test(NAME Exp002 COMMAND tso_exp002)
add_test(NAME Exp003 COMMAND tso_exp003)
add_test(NAME Exp004 COMMAND tso_exp004)
