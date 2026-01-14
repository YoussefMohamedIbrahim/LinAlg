cmake_minimum_required(VERSION 3.14)
project(LinAlg VERSION 1.0 LANGUAGES CXX)

add_library(LinAlg INTERFACE)

target_compile_features(LinAlg INTERFACE cxx_std_20)

target_include_directories(LinAlg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS LinAlg EXPORT LinAlgTargets INCLUDES DESTINATION include)
install(DIRECTORY include/linalg DESTINATION include)
install(EXPORT LinAlgTargets FILE LinAlgTargets.cmake NAMESPACE LinAlg:: DESTINATION lib/cmake/LinAlg)