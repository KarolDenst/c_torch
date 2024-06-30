cmake_minimum_required(VERSION 3.10)
project(TensorC)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
file(GLOB_RECURSE SOURCE_FILES ${SRC_DIR}/*.cpp)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*main.cpp$")

add_executable(TensorC src/main.cpp ${SOURCE_FILES})
target_include_directories(TensorC PUBLIC "${PROJECT_SOURCE_DIR}/src/tensor")
add_subdirectory(tests)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_custom_command(TARGET TensorC POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
)
