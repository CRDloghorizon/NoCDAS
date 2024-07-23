cmake_minimum_required(VERSION 3.10)
project(NoCDAS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

# Add source files, including .cpp files from both src and include directories
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/NoC/*.cpp")

# Add header files, including .hpp files from both src and include directories
file(GLOB_RECURSE HEADERS "src/*.hpp" "src/NoC/*.hpp")

# Add the executable
add_executable(NoCDASim ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(NoCDASim PRIVATE src src/NoC src/input)
