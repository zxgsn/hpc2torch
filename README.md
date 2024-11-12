# hpc2Torch
这个仓库打算搭建一个高性能底层库的测试框架，将会针对onnx的算子编写相关的高性能kernel，作为pytorch的补充，从python端对比手写kernel和pytorch库函数的性能以及精度对比。

## src
这个文件夹下面存放的是不同算子的kernel

## test
这个文件夹存放的是不同算子的python测试脚本，其中performance.py是功能文件，用于对比性能

## run.sh
运行仓库命令是：

bash run.sh

编译结束以后，可以直接做python端测试