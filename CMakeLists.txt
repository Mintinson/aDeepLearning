cmake_minimum_required(VERSION 3.20)

project(deep_learning VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CTest)

option(METANN_BUILD_EXAMPLES "Build metann examples" ON)
option(METANN_CLANG_USE_LIBCXX "Use libc++ and experimental library with Clang" ON)
option(METANN_BUILD_EXPERIMENTAL_TESTS "Build experimental/in-progress tests" ON)

add_library(metann INTERFACE)
add_library(metann::metann ALIAS metann)

target_compile_features(metann INTERFACE cxx_std_20)
target_compile_options(metann
    INTERFACE
    "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)

if(METANN_CLANG_USE_LIBCXX)
    target_compile_options(metann
        INTERFACE
        "$<$<CXX_COMPILER_ID:Clang>:-stdlib=libc++>"
        "$<$<CXX_COMPILER_ID:Clang>:-fexperimental-library>"
    )
    target_link_options(metann
        INTERFACE
        "$<$<CXX_COMPILER_ID:Clang>:-stdlib=libc++>"
    )
endif()

file(GLOB_RECURSE METANN_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/metann/*.hpp"
)

# Add headers for IDE discovery and source browsing.
target_sources(metann INTERFACE "$<BUILD_INTERFACE:${METANN_HEADERS}>")

target_include_directories(metann
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS metann EXPORT metannTargets)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/metann"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT metannTargets
    FILE metannTargets.cmake
    NAMESPACE metann::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/metann"
)

if(BUILD_TESTING)
    add_subdirectory(Tests)
endif()

if(METANN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# include_directories(utils)
# include_directories(policy)
# include_directories(data)

# add_subdirectory(chapter4)
# add_subdirectory(chapter5)
# add_subdirectory(chapter6)
# add_subdirectory(chapter7)
# add_subdirectory(chapter8)
