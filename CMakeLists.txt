cmake_minimum_required(VERSION 3.18)

project(MyCUDAProject LANGUAGES CXX CUDA)

# 查找 Python 库
find_package(Python3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# 设置 CUDA 编译选项
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# 添加选项控制不同的编译方式
option(USE_CUDA "Enable CUDA compilation" OFF)
option(USE_BANG "Enable BANG compilation" OFF)
option(USE_CPU "Enable CPU-only compilation" OFF)

# 查找源文件
file(GLOB CPP_SOURCE_FILES "src/**/*.cpp")
file(GLOB CUDA_SOURCE_FILES "src/**/*.cu")
file(GLOB BANG_SOURCE_FILES "src/**/*.mlu")

# 根据选项决定编译哪些源文件
if(USE_CUDA)
    message(STATUS "CUDA build enabled.")
    list(APPEND ALL_SOURCE_FILES ${CUDA_SOURCE_FILES} ${CPP_SOURCE_FILES})
elseif(USE_BANG)
    message(STATUS "BANG build enabled.")
    list(APPEND ALL_SOURCE_FILES ${BANG_SOURCE_FILES} ${CPP_SOURCE_FILES})
elseif(USE_CPU)
    message(STATUS "CPU-only build enabled.")
    list(APPEND ALL_SOURCE_FILES ${CPP_SOURCE_FILES})
else()
    message(FATAL_ERROR "No valid compilation mode specified. Please enable USE_CUDA, USE_BANG, or USE_CPU.")
endif()

# 创建库或可执行文件
add_library(my_library SHARED ${ALL_SOURCE_FILES})

# 设置编译选项
target_compile_features(my_library PUBLIC cxx_std_11)

# 链接 Python 库
target_link_libraries(my_library PRIVATE ${Python3_LIBRARIES})

# 指定输出目录
set_target_properties(my_library PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
