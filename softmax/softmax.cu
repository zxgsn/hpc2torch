#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <cub/block/block_reduce.cuh>

namespace py = pybind11;

struct __align__(8) DataMaxSum
{                  // update the global max and sum, store the
                   // output at max_tmp and sum_tmp
    float max_tmp; // store max
    float sum_tmp; // store sum
};
__device__ __forceinline__ DataMaxSum reduce_dms_op(DataMaxSum a,
                                                    DataMaxSum b)
{
    bool a_bigger = (a.max_tmp > b.max_tmp);
    DataMaxSum bigger = a_bigger ? a : b;
    DataMaxSum smaller = a_bigger ? b : a;
    bigger.sum_tmp = bigger.sum_tmp +
                     smaller.sum_tmp * __expf(smaller.max_tmp - bigger.max_tmp);

    return bigger;
}
template <typename T, int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__ void _blockSoftmaxKernel(
    T *__restrict input, T *__restrict output, int size, int dimsize,
    int stride)
{ // if set axis = 1, inputShape=[I,J,K,S]
  // tid = i(JKS) + j(KS) + k(S) + s

    // blockDim.x = size/dimsize = IKS
    // blockIdx.x = i(KS) + k(S) + s,blockIdx.x%stride = k(S) + s

    int tid =
        blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) *
                                  dimsize; // now, tid = i(JKS) + k(S) + s;

    DataMaxSum dms_partial;
    dms_partial.max_tmp = -__FLT_MAX__;
    dms_partial.sum_tmp = 0.0f;
    DataMaxSum dms_input;
    int remain = dimsize % BLOCK_DIM;
    int step = (dimsize - remain) / BLOCK_DIM + 1; // step <= numPerThread

    if (threadIdx.x < remain)
    {
        for (int ind = 0; ind < step; ind++)
        {
            dms_input.max_tmp =
                input[tid + (threadIdx.x * step + ind) * stride];

            dms_input.sum_tmp = 1.0f;
            dms_partial =
                reduce_dms_op(dms_partial,
                              dms_input); // reduce the data to one block
        }
    }
    else
    {
        for (int ind = 0; ind < step - 1; ind++)
        {
            dms_input.max_tmp =
                input[tid + (remain * step +
                             (threadIdx.x - remain) * (step - 1) + ind) *
                                stride];

            dms_input.sum_tmp = 1.0f;
            dms_partial =
                reduce_dms_op(dms_partial,
                              dms_input); // reduce the data to one block
        }
    }

    typedef cub::BlockReduce<DataMaxSum, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ DataMaxSum dms_total;
    DataMaxSum dms_block =
        BlockReduce(temp_storage).Reduce(dms_partial, reduce_dms_op);
    if (threadIdx.x ==
        0)
    { // must set threadIdx.x = 0 write the output to memory
        dms_total = dms_block;
    }
    __syncthreads();
    //-----------------
    if (threadIdx.x < remain)
    {
        for (int ind = 0; ind < step; ind++)
        {

            output[tid + (threadIdx.x * step + ind) * stride] =
                __expf(static_cast<float>(
                           input[tid + (threadIdx.x * step + ind) * stride]) -
                       dms_total.max_tmp) *
                __fdividef(1.0F, dms_total.sum_tmp);
        }
    }
    else
    {
        for (int ind = 0; ind < step - 1; ind++)
        {

            output[tid +
                   (remain * step + (threadIdx.x - remain) * (step - 1) + ind) *
                       stride] =
                __expf(static_cast<float>(
                           input[tid +
                                 (remain * step +
                                  (threadIdx.x - remain) * (step - 1) + ind) *
                                     stride]) -
                       dms_total.max_tmp) *
                __fdividef(1.0F, dms_total.sum_tmp);
        }
    }
}

template <typename T, int BLOCK_DIM, int numPerThread>
__global__ void
_blockSoftmaxKernel(T *__restrict input, T *__restrict output, int size,
                    int dimsize,
                    int stride)
{ // if set axis = 1, inputShape=[I,J,K,S]
  // tid = i(JKS) + j(KS) + k(S) + s

    // blockDim.x = size/dimsize = IKS
    // blockIdx.x = i(KS) + k(S) + s,blockIdx.x%stride = k(S) + s

    int tid =
        blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) *
                                  dimsize; // now, tid = i(JKS) + k(S) + s;
    int remain = dimsize % BLOCK_DIM;
    int step = (dimsize - remain) / BLOCK_DIM + 1; // step <= numPerThread
    float dataPerThread[numPerThread];

    DataMaxSum dms_partial;
    dms_partial.max_tmp = -__FLT_MAX__;
    dms_partial.sum_tmp = 0.0f;
    DataMaxSum dms_input;
    if (threadIdx.x < remain)
    {
        for (int ind = 0; ind < step; ind++)
        {
            dataPerThread[ind] =
                input[tid + (threadIdx.x * step + ind) * stride];
            dms_input.max_tmp = dataPerThread[ind];
            dms_input.sum_tmp = 1.0f;
            dms_partial =
                reduce_dms_op(dms_partial,
                              dms_input); // reduce the data to one block
        }
    }
    else
    {
        for (int ind = 0; ind < step - 1; ind++)
        {
            dataPerThread[ind] =
                input[tid + (remain * step +
                             (threadIdx.x - remain) * (step - 1) + ind) *
                                stride];
            dms_input.max_tmp = dataPerThread[ind];
            dms_input.sum_tmp = 1.0f;
            dms_partial =
                reduce_dms_op(dms_partial,
                              dms_input); // reduce the data to one block
        }
    }

    typedef cub::BlockReduce<DataMaxSum, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ DataMaxSum dms_total;
    DataMaxSum dms_block =
        BlockReduce(temp_storage).Reduce(dms_partial, reduce_dms_op);
    if (threadIdx.x ==
        0)
    { // must set threadIdx.x = 0 write the output to memory
        dms_total = dms_block;
    }
    __syncthreads();
    //-----------------
    if (threadIdx.x < remain)
    {
        for (int ind = 0; ind < step; ind++)
        {
            output[tid + (threadIdx.x * step + ind) * stride] =
                __expf(dataPerThread[ind] - dms_total.max_tmp) *
                __fdividef(1.0F, dms_total.sum_tmp);
        }
    }
    else
    {
        for (int ind = 0; ind < step - 1; ind++)
        {
            output[tid +
                   (remain * step + (threadIdx.x - remain) * (step - 1) + ind) *
                       stride] =
                __expf(dataPerThread[ind] - dms_total.max_tmp) *
                __fdividef(1.0F, dms_total.sum_tmp);
        }
    }
}

template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return max(a, b);
    }
};
template <template <typename> class ReductionOp, typename T,
          int thread_group_width>
__inline__ __device__ T WarpAllReduce(T val)
{
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <typename T, int BLOCK_DIM_x, int BLOCK_DIM_y, int numPerThreadx>
__global__ void _warpSoftmaxKernel(T *__restrict input, T *__restrict output,
                                   int size, int dimsize, int stride)
{
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int otherSize = size / dimsize;
    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;
    float dataPerThreadx[numPerThreadx];
    if (otherIdx < otherSize)
    {

        __shared__ float max_total[BLOCK_DIM_y];
        __shared__ float sum_total[BLOCK_DIM_y];
        float max_data = -__FLT_MAX__;

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++)
        {
            dataPerThreadx[ph] =
                input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride];
            max_data = max(max_data, dataPerThreadx[ph]);
        }

        max_data = WarpAllReduce<MaxOp, float, BLOCK_DIM_x>(max_data);

        if (threadIdx.x == 0)
            max_total[threadIdx.y] = max_data;

        //--------------------------------------------
        float sum_data = 0.0f;

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++)
        {
            dataPerThreadx[ph] =
                __expf(dataPerThreadx[ph] - max_total[threadIdx.y]);
            sum_data += dataPerThreadx[ph];
        }

        sum_data = WarpAllReduce<SumOp, float, BLOCK_DIM_x>(sum_data);

        if (threadIdx.x == 0)
            sum_total[threadIdx.y] = sum_data;

        //--------------------------------------------

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++)
        {
            output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                dataPerThreadx[ph] * __fdividef(1.0F, sum_total[threadIdx.y]);
        }
    }
}
void softmaxLaunch(torch::Tensor input_tensor, torch::Tensor output_tensor, int size, int dimsize, int stride)
{
    // 确保输入和输出张量都是在CUDA上
    TORCH_CHECK(input_tensor.is_cuda(), "Input tensor must be on the GPU");
    TORCH_CHECK(output_tensor.is_cuda(), "Output tensor must be on the GPU");

    float *input = input_tensor.data_ptr<float>();
    float *output = output_tensor.data_ptr<float>();
    // 计算结束以后结果会自动更新到output_tensor，不需要额外复制

    int num_blocks = size / dimsize;

    if (dimsize > 1024 * 128)
    {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024>
            <<<num_blocks, BLOCK_DIM>>>(input, output, size, dimsize, stride);
    }
    else if (dimsize > 1024 * 64)
    {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024, 128>
            <<<num_blocks, BLOCK_DIM>>>(input, output, size, dimsize, stride);
    }
    else if (dimsize > 1024 * 32)
    {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024, 64>
            <<<num_blocks, BLOCK_DIM>>>(input, output, size, dimsize, stride);
    }
    else if (dimsize > 1024 * 16)
    {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024, 32>
            <<<num_blocks, BLOCK_DIM>>>(input, output, size, dimsize, stride);
    }
    else if (dimsize > 1024 * 4)
    {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024, 16>
            <<<num_blocks, BLOCK_DIM>>>(input, output, size, dimsize, stride);
    }
    else if (dimsize > 1024)
    {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024, 4>
            <<<num_blocks, BLOCK_DIM>>>(input, output, size, dimsize, stride);
    }
    else if (dimsize > 31)
    {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<float, 32, 32, 32>
            <<<grid_dim, block_dim>>>(input, output, size, dimsize, stride);
    }
    else if (dimsize > 15)
    {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<float, 16, 64, 2>
            <<<grid_dim, block_dim>>>(input, output, size, dimsize, stride);
    }
    else if (dimsize > 7)
    {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<float, 8, 128, 2>
            <<<grid_dim, block_dim>>>(input, output, size, dimsize, stride);
    }
    else
    {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<float, 4, 256, 2>
            <<<grid_dim, block_dim>>>(input, output, size, dimsize, stride);
    }
    cudaDeviceSynchronize();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // 第一个参数"softmax"表示注册到python模块中的函数名称，可以替换为其他名字，使用方法为：模块.softmax
    // 第二个参数softmaxLaunch是上面编写的kernel launch 函数，这里需要获得该函数的地址
    // 第三个参数"Cuda Core softmax function"是描述性文字，可以修改
    // 后面的py::arg是用来为softmax定义参数的，这些参数的数目，顺序必须和softmaxLaunch保持一致，为了增加可读性，最好名字也一致
    m.def("softmax", &softmaxLaunch, "Cuda Core softmax function",
          py::arg("input"), py::arg("output"), py::arg("size"), py::arg("dimsize"), py::arg("stride"));
}
