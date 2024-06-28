cmake_minimum_required(VERSION 3.10)

# Set the project name
project(TensorC)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(TensorC src/main.cpp src/tensor/tensor.cpp)

# Include directories
target_include_directories(TensorC PUBLIC "${PROJECT_SOURCE_DIR}/src/tensor")

# Add tests directory
add_subdirectory(tests)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Copy compile_commands.json to project root
add_custom_command(TARGET TensorC POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
)
