#include <torch/extension.h>

// 声明在 softmax.cu 和 attention.cu 中的函数
void softmaxLaunch(torch::Tensor input, torch::Tensor output, int size, int dimsize, int stride);
void attentionLaunch(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int N, int d, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      // 第一个参数"softmax"表示注册到python模块中的函数名称，可以替换为其他名字，使用方法为：模块.softmax
      // 第二个参数softmaxLaunch是上面编写的kernel launch 函数，这里需要获得该函数的地址
      // 第三个参数"Cuda Core softmax function"是描述性文字，可以修改
      // 后面的py::arg是用来为softmax定义参数的，这些参数的数目，顺序必须和softmaxLaunch保持一致，为了增加可读性，最好名字也一致
      m.def("softmax", &softmaxLaunch, "Cuda Core softmax function",
            py::arg("input"), py::arg("output"), py::arg("size"), py::arg("dimsize"), py::arg("stride"));

      m.def("attention", &attentionLaunch, "Cuda Core attention function",
            py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("N"), py::arg("d"), py::arg("output"));
}
