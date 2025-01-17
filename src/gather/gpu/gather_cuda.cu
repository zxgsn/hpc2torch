#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cassert>

template <typename T>
__global__ void gather_kernel(const T* input, const int64_t* indices, T* output, int64_t axis_size, int64_t outer_size, int64_t inner_size, int64_t index_size, int64_t axis) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    int thread_id = threadIdx.x;  // 线程在当前线程块中的索引
    index_size = static_cast<int>(index_size);
    // printf("axis: %d, inner: %ld, outer: %ld, index: %ld\n", axis, inner_size, outer_size, index_size);
    // 计算总线程数
    int64_t total = outer_size * inner_size * index_size;
    // printf("BlockIdx: %d, blockDim: %d, Thread ID: %ld, idx: %ld, total: %ld, axis_Size: %ld\n", blockIdx.x, blockDim.x, thread_id, idx, total, axis_size);
    if (idx < total) {
        // 重新计算 outer_idx 和 inner_idx 以适应 axis=0 的情形
        int64_t outer_idx = idx / (inner_size * index_size);

        int64_t index_idx = (idx % (index_size * inner_size)) / inner_size;
        
        int64_t inner_idx = idx % inner_size;
        
        int64_t index;

        // 确保索引有效
        // if (index >= 0) {
            // 计算output的线性索引并赋值
            int64_t input_idx;
            if (axis == 0)
            {
                // printf("v = %ld\n", outer_idx * inner_size + index_idx);
                index = indices[outer_idx * index_size + index_idx]; // 结合 outer_idx 和 inner_idx 获取索引
                // axis=0时沿第一维gather
                input_idx = index * inner_size + inner_idx;
            }
            else
            {
                // axis=1时沿第二维gather
                index = indices[index_idx];
                input_idx = outer_idx * axis_size + index;
            }
            // printf("idx = %ld, outer_idx = %ld, inner_idx = %ld, index = %ld, input_idx = %ld\n", idx, outer_idx, inner_idx, index, input_idx);
            output[idx] = input[input_idx];
        // }
    }
}



// 主机函数：调用 CUDA 核函数
template <typename T>
void gather_cuda(const T* input, const int64_t* indices, T* output, int64_t axis_size, int64_t outer_size, int64_t inner_size, int64_t index_size, int64_t axis) {

    // 定义线程块和网格大小
    int blockSize = 64;
    int total = (index_size * outer_size * inner_size + blockSize - 1);
    int gridSize = static_cast<int>(total / blockSize);
    // printf("blocksize = %d,  gridsize = %d, total = %d, index = %d,  outer = %d , inner = %d\n", blockSize, gridSize, (index_size * outer_size * inner_size + blockSize - 1), index_size, outer_size, inner_size);

    // 调用核函数
    gather_kernel<T><<<gridSize, blockSize>>>(input, indices, output, axis_size, outer_size, inner_size, index_size, axis);

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    // cudaDeviceSynchronize();
}

// 显式实例化模板函数，支持 ONNX 的所有数据类型
template void gather_cuda<float>(const float *, const int64_t *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);
template void gather_cuda<half>(const half *, const int64_t *, half *, int64_t, int64_t, int64_t, int64_t, int64_t);

// 导出 C 接口
extern "C" {
    void gather_cuda_float(const float *input, const int64_t *indices, float *output, int64_t axis_size, int64_t outer_size, int64_t inner_size, int64_t index_size, int64_t axis)
    {
        gather_cuda<float>(input, indices, output, axis_size, outer_size, inner_size, index_size, axis);
    }

    void gather_cuda_half(const half *input, const int64_t *indices, half *output, int64_t axis_size, int64_t outer_size, int64_t inner_size, int64_t index_size, int64_t axis)
    {
        gather_cuda<half>(input, indices, output, axis_size, outer_size, inner_size, index_size, axis);
    }
}