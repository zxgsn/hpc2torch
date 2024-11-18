#include <stdio.h>
#include <math.h>
extern "C" void softmax_cpu_f32(float *input, float *output, int size, int dimsize, int stride)
{
    int othersize = size / dimsize;
    for (int ind = 0; ind < othersize; ind++)
    {                                                            // ind = i(KS) + k(S) + s
        int tid = ind % stride + (ind - ind % stride) * dimsize; // now, tid = i(JKS) + k(S) + s;
        float localM = -__FLT_MAX__;
        for (int j = 0; j < dimsize; j++)
        {
            int index = tid + j * stride;
            localM = fmax(localM, input[index]);
        }
        float localS = 0.0f;
        for (int j = 0; j < dimsize; j++)
        {
            int index = tid + j * stride;
            localS += exp(input[index] - localM);
        }
        for (int j = 0; j < dimsize; j++)
        {
            int index = tid + j * stride;
            output[index] = exp(input[index] - localM) / localS;
        }
    }
}