#include <cublas_v2.h>
extern "C" void matmul_cudnn_f32(void const *dA, void const *dB, void *dC, int M, int K, int N)
{
    cublasHandle_t handle; // cublas句柄
    cublasCreate(&handle); // 初始化句柄
    float alpha = 1.0;
    float beta = 0.0;

    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, dA, K, dB, N, &beta, dC, M);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 dB, CUDA_R_32F, N,
                 dA, CUDA_R_32F, K,
                 &beta,
                 dC, CUDA_R_32F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
