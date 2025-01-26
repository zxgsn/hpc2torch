#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cassert>

constexpr int BLOCK_DIM_X = 128;
constexpr int BLOCK_DIM_Y = 1;
constexpr int ITEMS_PER_THREAD = 2; // 每个线程处理多个元素

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
    const int64_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // 每个线程处理ITEMS_PER_THREAD个元素
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        const int64_t curr_idx = x_idx * ITEMS_PER_THREAD + i;
        if (curr_idx >= inner_size * index_size)
            break;

        const int64_t index_idx = curr_idx / inner_size;
        const int64_t inner_idx = curr_idx % inner_size;

        const int64_t index = indices[index_idx];

        const int64_t input_idx = (y_idx * axis_size + index) * inner_size + inner_idx;
        const int64_t out_idx = y_idx * (index_size * inner_size) + curr_idx;

        output[out_idx] = input[input_idx];
    }
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
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid_dim(
        ceil_div(inner_size * index_size, BLOCK_DIM_X * ITEMS_PER_THREAD),
        ceil_div(outer_size, BLOCK_DIM_Y));

    gather_kernel<T><<<grid_dim, block_dim>>>(
        input, indices, output,
        axis_size, outer_size, inner_size,
        index_size, axis);
}

template void gather_cuda<float>(const float *, const int64_t *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);
template void gather_cuda<half>(const half *, const int64_t *, half *, int64_t, int64_t, int64_t, int64_t, int64_t);

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