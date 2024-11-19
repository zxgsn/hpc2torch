#include <stdio.h>
#include <math.h>

extern "C" void layernorm_cpu_f32(float const *input, float const *scale, float const *bias, float *output, float eps, int size, int behindsize)
{
}