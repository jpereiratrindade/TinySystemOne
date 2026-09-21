cmake_minimum_required(VERSION 3.25)
project(TinySystemOne
    VERSION 0.5.0
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

add_executable(tso_exp005 src/exp005_main.cpp)
target_include_directories(tso_exp005 PRIVATE include)

add_executable(tso_exp006 src/exp006_main.cpp)
target_include_directories(tso_exp006 PRIVATE include)

add_executable(tso_exp007 src/exp007_main.cpp)
target_include_directories(tso_exp007 PRIVATE include)

add_executable(tso_exp008 src/exp008_main.cpp)
target_include_directories(tso_exp008 PRIVATE include)

add_executable(tso_exp009 src/exp009_main.cpp)
target_include_directories(tso_exp009 PRIVATE include)

# High-Level Production Classifier CLI & Tool
add_executable(tso_classifier src/classifier_cli.cpp)
target_include_directories(tso_classifier PRIVATE include)

# Unit & Acceptance Tests Targets
add_executable(test_math tests/test_math.cpp)
target_include_directories(test_math PRIVATE include)

add_executable(test_dataset tests/test_dataset.cpp)
target_include_directories(test_dataset PRIVATE include)

add_executable(test_acceptance tests/test_acceptance.cpp)
target_include_directories(test_acceptance PRIVATE include)

add_executable(test_classifier tests/test_classifier.cpp)
target_include_directories(test_classifier PRIVATE include)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_include_directories(test_tokenizer PRIVATE include)

add_executable(test_text_tokenizer tests/test_text_tokenizer.cpp)
target_include_directories(test_text_tokenizer PRIVATE include)

add_executable(test_attention tests/test_attention.cpp)
target_include_directories(test_attention PRIVATE include)

enable_testing()
add_test(NAME TestMath COMMAND test_math)
add_test(NAME TestDataset COMMAND test_dataset)
add_test(NAME TestAcceptance COMMAND test_acceptance)
add_test(NAME TestClassifier COMMAND test_classifier)
add_test(NAME TestTokenizer COMMAND test_tokenizer)
add_test(NAME TestTextTokenizer COMMAND test_text_tokenizer)
add_test(NAME TestAttention COMMAND test_attention)
add_test(NAME Exp001 COMMAND tso_exp001)
add_test(NAME Exp002 COMMAND tso_exp002)
add_test(NAME Exp003 COMMAND tso_exp003)
add_test(NAME Exp004 COMMAND tso_exp004)
add_test(NAME Exp005 COMMAND tso_exp005)
add_test(NAME Exp006 COMMAND tso_exp006)
add_test(NAME Exp007 COMMAND tso_exp007)
add_test(NAME Exp008 COMMAND tso_exp008)
add_test(NAME Exp009 COMMAND tso_exp009)



