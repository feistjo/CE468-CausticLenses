#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <cutil.h>

#include "util.h"
#include "kernels.h"

#define KERN_1D(nthreads) ((nthreads) + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE

// __global__ void dev_sum_reduce(float *in, float *out) {

//     __shared__ float shared[BLK_SIZE];
    
//     unsigned tid = threadIdx.x;
//     unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

//     shared[tid] = in[i];
    
//     __syncthreads();
    
//     for (unsigned s = 1; s < blockDim.x; s *= 2) {
//         if (tid % (2 * s) == 0) {

//             shared[tid] += shared[tid + s];
//         }
//         __syncthreads();
//     }

//     if (tid == 0) out[blockIdx.x] = shared[0];
// }

// float sum_reduce(Matrix m) {
//     unsigned len = m.wid * m.hgt;

//     float sum = 0;

//     float *part_sums;
//     cudaMalloc(&part_sums, BLK_SIZE * sizeof(float));
//     dev_sum_reduce<<<KERN_1D(len)>>>(m.elems, part_sums);
//     dev_sum_reduce<<<1, BLK_SIZE>>>(part_sums, part_sums);
//     cudaDeviceSynchronize();

//     cudaMemcpy(&sum, part_sums, sizeof(float), cudaMemcpyDeviceToHost);
//     cudaFree(part_sums);
//     return sum;
// }