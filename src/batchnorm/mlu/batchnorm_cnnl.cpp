#include "cnnl.h"
#include "cnnl_extra.h"
#include <vector>

template<typename T>
void batchnormCnnlDevice(void const *input, void const *scale, void const *bias, void const *mean, void const *var, void *output, int *shape, int nDim, float eps, cnnlHandle_t &handle, cnrtQueue_t &queue)
{
    std::vector<int> inDim(nDim);//原始input的形状为[n,c,h,w]
    std::vector<int> outDim(nDim);
    std::vector<int> fbmvDim(1);//batchnorm描述scale,bias,mean,var的desc
    int size = 1;
    for (int i = 0; i < nDim; i++) {
        inDim[i] = shape[i];
        outDim[i] = shape[i];
        size *= shape[i];
    }
    fbmvDim[0] = shape[nDim - 1];
    //下面开始针对[n,h,w,c]的tmpGdramI做batchnorm变换
    cnnlTensorDescriptor_t x_desc, z_desc, filter_bias_mean_var_desc;
    cnnlCreateTensorDescriptor(&x_desc);
    cnnlCreateTensorDescriptor(&z_desc);
    cnnlCreateTensorDescriptor(&filter_bias_mean_var_desc);


    
    cnnlTensorLayout_t layout;
    if(nDim == 2){
        layout = CNNL_LAYOUT_NC;
    }
    else if(nDim == 3){
        layout = CNNL_LAYOUT_NLC;
    }
    else if(nDim == 4){
        layout = CNNL_LAYOUT_NHWC;
    }
    else if(nDim == 5){
        layout = CNNL_LAYOUT_NDHWC;
    }
    if(sizeof(T) == 2){
        cnnlSetTensorDescriptor(
            x_desc, layout, CNNL_DTYPE_HALF,
            inDim.size(), inDim.data());
        cnnlSetTensorDescriptor(
            z_desc, layout, CNNL_DTYPE_HALF,
            outDim.size(), outDim.data());
        cnnlSetTensorDescriptor(
            filter_bias_mean_var_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            fbmvDim.size(), fbmvDim.data());
    }
    else if(sizeof(T) == 4){
        cnnlSetTensorDescriptor(
            x_desc, layout, CNNL_DTYPE_FLOAT,
            inDim.size(), inDim.data());
        cnnlSetTensorDescriptor(
            z_desc, layout, CNNL_DTYPE_FLOAT,
            outDim.size(), outDim.data());
        cnnlSetTensorDescriptor(
            filter_bias_mean_var_desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            fbmvDim.size(), fbmvDim.data());
    }

    //下面开始针对input做转置，nchw2nhwc
    T *tmpGdramI, *tmpGdramO;//batchnorm库函数只能处理[n,h,w,c],tmpGdramI作为转置来变换input
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramI, size * sizeof(T)));
    CNRT_CHECK(cnrtMalloc((void **)&tmpGdramO, size * sizeof(T)));
    cnnlTensorDescriptor_t aDesc, cDesc;
       
    cnnlCreateTensorDescriptor(&aDesc);
    cnnlCreateTensorDescriptor(&cDesc);
    if(sizeof(T) == 2){
        cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            inDim.size(), inDim.data());
        
        cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
            outDim.size(), outDim.data());
    }
    else if(sizeof(T) == 4){
        cnnlSetTensorDescriptor(
            aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            inDim.size(), inDim.data());
        
        cnnlSetTensorDescriptor(
            cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
            outDim.size(), outDim.data());
    }
    if(nDim > 2){
        cnnlTransposeDescriptor_t desc;
        cnnlCreateTransposeDescriptor(&desc);
        
        std::vector<int> permuteI(nDim);
        std::vector<int> permuteO(nDim);
        for (int i = 0; i < nDim; i++) {
            permuteI[i] = i;
            permuteO[i] = i;
        }
        if(nDim > 2){
            for (int i = 0; i < nDim; i++) {
                if(i >= 1){
                    permuteI[i] = i + 1;
                }
                if(i >= 2){
                    permuteO[i] = i - 1;
                }
            }
            permuteI[nDim - 1] = 1;
            permuteO[1] = nDim - 1;
        }
        
        cnnlSetTransposeDescriptor(desc, nDim, permuteI.data());
        size_t tSize;
        cnnlGetTransposeWorkspaceSize(handle, aDesc, desc, &tSize);
        void *workspace;
        cnrtMalloc(&workspace, tSize);
        cnnlTranspose_v2(handle, desc, aDesc, input, cDesc,
                                tmpGdramI, workspace, tSize);
        CNRT_CHECK(cnrtQueueSync(queue));                         
        //上面成功对input做好了nchw2nhwc，下面开始正式计算batchnorm
        cnnlBatchNormForwardInference(handle, nullptr, nullptr, x_desc, tmpGdramI, 
                                filter_bias_mean_var_desc, scale, bias, mean, var, 
                                eps, z_desc, tmpGdramO);
        CNRT_CHECK(cnrtQueueSync(queue));
        //下面开始提前对output做转置：nhwc2nchw
        cnnlTranspose_v2(handle, desc, aDesc, tmpGdramO, cDesc,
                                output, workspace, tSize);
        CNRT_CHECK(cnrtQueueSync(queue));  
        cnrtFree(tmpGdramI);
        cnrtFree(tmpGdramO);
        cnnlDestroyTensorDescriptor(aDesc);
        cnnlDestroyTensorDescriptor(cDesc);
        cnnlDestroyTransposeDescriptor(desc);
    }
    else if (nDim == 2){
        cnnlBatchNormForwardInference(handle, nullptr, nullptr, x_desc, input, 
                                filter_bias_mean_var_desc, scale, bias, mean, var, 
                                eps, z_desc, output);
        CNRT_CHECK(cnrtQueueSync(queue));
    }
    

    cnnlDestroyTensorDescriptor(x_desc);
    cnnlDestroyTensorDescriptor(z_desc);
    cnnlDestroyTensorDescriptor(filter_bias_mean_var_desc);
    
}
template<typename T>
void batchnormCnnl(void const *input, void const *scale, void const *bias, void const *mean, void const *var, void *output, int *shape, int nDim, float eps)
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnnlHandle_t handle;
    cnnlCreate(&handle);
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnnlSetQueue(handle, queue); // 将队列绑定到 handle 中, 此接口也可用来更改句柄中的队列。

    batchnormCnnlDevice<T>(input, scale, bias, mean, var, output, shape, nDim, eps, handle, queue);
    
    cnnlDestroy(handle);
    CNRT_CHECK(cnrtQueueDestroy(queue));

    
}
extern "C" void batchnorm_cnnl_f32(void const *input, void const *scale, void const *bias, void const *mean, void const *var, void *output, int *shape, int nDim, float eps){
    batchnormCnnl<float>(input, scale, bias, mean, var, output, shape, nDim, eps);
}
extern "C" void batchnorm_cnnl_f16(void const *input, void const *scale, void const *bias, void const *mean, void const *var, void *output, int *shape, int nDim, float eps){
    batchnormCnnl<uint16_t>(input, scale, bias, mean, var, output, shape, nDim, eps);
}





