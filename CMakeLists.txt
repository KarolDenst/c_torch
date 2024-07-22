cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(CTorch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
file(GLOB_RECURSE SOURCE_FILES ${SRC_DIR}/*.cpp)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*main.cpp$")

add_executable(CTorch src/main.cpp ${SOURCE_FILES})
target_include_directories(CTorch PUBLIC "${PROJECT_SOURCE_DIR}/src/tensor")
add_subdirectory(tests)

add_custom_command(TARGET CTorch POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/compile_commands.json
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
)
