#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cassert>

// 常量定义
constexpr int BLOCK_DIM_X = 4;
constexpr int BLOCK_DIM_Y = 4;

// 辅助函数：计算对齐的网格尺寸
inline int64_t ceil_div(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

template <typename T>
__global__ void gather_kernel(const T *input,
                              const int64_t *indices,
                              T *output,
                              int64_t axis_size,
                              int64_t outer_size,
                              int64_t inner_size,
                              int64_t index_size,
                              int64_t axis)
{
    const int64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= (inner_size * index_size) || idx_y >= outer_size)
        return;

    const int64_t index_idx = idx_x / inner_size;
    const int64_t inner_idx = idx_x % inner_size;

    const int64_t index = indices[index_idx];
    const int64_t input_idx = (idx_y * axis_size + index) * inner_size + inner_idx;
    const int64_t out_idx = idx_y * (index_size * inner_size) + idx_x;

    output[out_idx] = input[input_idx];
}

template <typename T>
void gather_cuda(const T *input,
                 const int64_t *indices,
                 T *output,
                 int64_t axis_size,
                 int64_t outer_size,
                 int64_t inner_size,
                 int64_t index_size,
                 int64_t axis)
{

    // 设定2D线程块和网格维度
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid_dim(
        ceil_div(inner_size * index_size, BLOCK_DIM_X),
        ceil_div(outer_size, BLOCK_DIM_Y));

    // 启动核函数
    gather_kernel<T><<<grid_dim, block_dim>>>(
        input, indices, output,
        axis_size, outer_size, inner_size,
        index_size, axis);

    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // cudaDeviceSynchronize();
}

// 显式实例化模板函数
template void gather_cuda<float>(const float *, const int64_t *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);
template void gather_cuda<half>(const half *, const int64_t *, half *, int64_t, int64_t, int64_t, int64_t, int64_t);

// 导出C接口
extern "C"
{
    void gather_cuda_float(const float *input,
                           const int64_t *indices,
                           float *output,
                           int64_t axis_size,
                           int64_t outer_size,
                           int64_t inner_size,
                           int64_t index_size,
                           int64_t axis)
    {
        gather_cuda<float>(input, indices, output,
                           axis_size, outer_size, inner_size,
                           index_size, axis);
    }

    void gather_cuda_half(const half *input,
                          const int64_t *indices,
                          half *output,
                          int64_t axis_size,
                          int64_t outer_size,
                          int64_t inner_size,
                          int64_t index_size,
                          int64_t axis)
    {
        gather_cuda<half>(input, indices, output,
                          axis_size, outer_size, inner_size,
                          index_size, axis);
    }
}