cmake_minimum_required(VERSION 3.24)

project(convexity VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CONVEXITY_CXX_STANDARD 23)
else()
    set(CONVEXITY_CXX_STANDARD 20)
endif()

set(CMAKE_CXX_STANDARD ${CONVEXITY_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CONVEXITY_BUILD_CLIENT "Build the example C++ client" ON)
option(CONVEXITY_BUILD_TESTS "Build C++ tests" ON)

if(DEFINED SKBUILD)
    set(_convexity_build_python_default ON)
else()
    set(_convexity_build_python_default OFF)
endif()
option(CONVEXITY_BUILD_PYTHON "Build the Python bindings" ${_convexity_build_python_default})
option(CONVEXITY_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ConvexityWarnings)

add_library(convexity_core SHARED
    src/core/version.cpp
)
add_library(convexity::core ALIAS convexity_core)

target_compile_features(convexity_core PUBLIC cxx_std_${CONVEXITY_CXX_STANDARD})
target_include_directories(convexity_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(convexity_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME core
)
convexity_enable_warnings(convexity_core)
convexity_enable_sanitizers(convexity_core)

if(CONVEXITY_BUILD_CLIENT)
    add_executable(convexity_cpp_client apps/cpp_client/main.cpp)
    target_link_libraries(convexity_cpp_client PRIVATE convexity::core)
    convexity_enable_warnings(convexity_cpp_client)
    convexity_enable_sanitizers(convexity_cpp_client)
endif()

if(CONVEXITY_BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)

    add_executable(convexity_cpp_tests tests/cpp/version_test.cpp)
    target_link_libraries(convexity_cpp_tests
        PRIVATE
            convexity::core
            GTest::gtest_main
    )
    convexity_enable_warnings(convexity_cpp_tests)
    convexity_enable_sanitizers(convexity_cpp_tests)

    include(GoogleTest)
    gtest_discover_tests(convexity_cpp_tests)
endif()

if(CONVEXITY_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(convexity_python MODULE bindings/python/module.cpp)
    target_link_libraries(convexity_python PRIVATE convexity::core)
    set_target_properties(convexity_python PROPERTIES OUTPUT_NAME _core)
    if(SKBUILD)
        set_target_properties(convexity_python PROPERTIES INSTALL_RPATH "$ORIGIN")
    else()
        set_target_properties(convexity_python PROPERTIES INSTALL_RPATH "$ORIGIN/../..")
    endif()
    convexity_enable_warnings(convexity_python)
    convexity_enable_sanitizers(convexity_python)

    if(SKBUILD)
        set(_convexity_python_install_dir "${SKBUILD_PLATLIB_DIR}/convexity")
    else()
        set(_convexity_python_install_dir "${CMAKE_INSTALL_LIBDIR}/python/convexity")
    endif()

    install(TARGETS convexity_python
        LIBRARY DESTINATION "${_convexity_python_install_dir}"
        RUNTIME DESTINATION "${_convexity_python_install_dir}"
    )
    install(FILES python/convexity/__init__.py
        DESTINATION "${_convexity_python_install_dir}"
    )
endif()

if(SKBUILD)
    set(_convexity_core_install_dir "${SKBUILD_PLATLIB_DIR}/convexity")
else()
    set(_convexity_core_install_dir "${CMAKE_INSTALL_LIBDIR}")
endif()

install(TARGETS convexity_core
    EXPORT convexityTargets
    LIBRARY DESTINATION "${_convexity_core_install_dir}"
    RUNTIME DESTINATION "${_convexity_core_install_dir}"
    ARCHIVE DESTINATION "${_convexity_core_install_dir}"
)
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_package_config_file(
    cmake/convexityConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/convexityConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/convexity"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/convexityConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT convexityTargets
    FILE convexityTargets.cmake
    NAMESPACE convexity::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/convexity"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/convexityConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/convexityConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/convexity"
)
