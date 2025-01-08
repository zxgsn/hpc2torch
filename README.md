# hpc2Torch

这个仓库打算搭建一个高性能底层库的测试框架，将会针对onnx的算子编写相关的高性能kernel，作为pytorch的补充，从python端对比手写kernel和pytorch库函数的性能以及精度对比。

## src

这个文件夹下面存放的是不同算子的kernel

## test

这个文件夹存放的是不同算子的python测试脚本，其中performance.py是功能文件，用于对比性能

## 测试

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
  ```
