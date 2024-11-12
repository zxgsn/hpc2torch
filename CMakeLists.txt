cmake_minimum_required(VERSION 3.18)

project(MyCUDAProject LANGUAGES CXX CUDA)

# 查找 Python 库
find_package(Python3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# 设置 CUDA 编译选项
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# 查找所有 .cu 文件
file(GLOB CUDA_SOURCE_FILES 
    "src/**/*.cu"  # 匹配 src 目录下的所有 .cu 文件
    "src/**/*.cpp"
)

# 添加 CUDA 可执行文件或库
add_library(my_cuda_library SHARED ${CUDA_SOURCE_FILES})

# 设置编译选项
target_compile_features(my_cuda_library PUBLIC cxx_std_11)

# 链接 Python 库
target_link_libraries(my_cuda_library PRIVATE ${Python3_LIBRARIES})

# 指定输出目录
set_target_properties(my_cuda_library PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
