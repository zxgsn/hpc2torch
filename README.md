# 2024 冬季 CUDA 作业

参照 [ONNX 的 Gather 算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather)实现一个 CUDA kernel。该 kernel 需要接收三个输入参数：`data`、`indices` 和 `axis`，并输出 `output`。`axis` 的值缺省为 0。在开始实现之前，建议先访问 cuDNN 的官方网站，查询是否有现成的库函数可用于实现 Gather 操作；如果存在现成的库函数，可以同时添加库函数以及手写 cuda 算子的实现作为对比。

1. fork 本仓库；
2. 在 `src/gather/gpu` 目录下添加名为 `gather_cuda.cu` 的文件，并在其中实现 Gather 算子；
3. Gather 需要支持 ONNX 官方文档中 `data` 的所有数据类型，并与 PyTorch 保持 0 误差；
4. 在框架的 `test/gather.py`，验证你编写的 Gather 算子的精度和性能，必要的规模包括：

    - `(3, 2), (2, 2), 0`
      > 表示 `data` 的形状为 `(3, 2)`，`indices` 的形状为 `(2, 2)`，且 `axis` 的值为 0；
    - `(3, 2), (1, 2), 1`；
    - `(50257, 768), (16, 1024), 0`；

5. 性能要求：与 PyTorch 的实现相比，你编写的 Gather 算子在 `(50257, 768), (16, 1024), 0` 的性能应至少提高 20%；
6. 完成后向原仓库提交一个 issue，标题为`提交作业`，内容包括你的仓库链接，以及可展示你测试的环境、规模和性能差距的文本或图片；

## 命令

- 编译项目

  ```shell
  make
  ```

- 清理项目

  ```shell
  make clean
  ```

- 测试算子

  ```shell
  make test
  ```
