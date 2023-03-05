#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cutil.h>

#include "util.h"

Matrix init_matrix(const unsigned hgt, const unsigned wid) {
    unsigned n_bytes = hgt * wid * sizeof(float);

    Matrix out;
    out.hgt = hgt;
    out.wid = wid;

    cudaMalloc((void **)&out.elems, n_bytes);
    cudaMemset(out.elems, 0, n_bytes);

    return out;
}

Matrix to_device(const Matrix m) {
    Matrix m_d = m;
    size_t len = m.wid * m.hgt * sizeof(float);
    cudaMalloc((void **)&m_d.elems, len);
    cudaMemcpy(m_d.elems, m.elems, len, cudaMemcpyHostToDevice);
    return m_d;
}

__global__ void dev_init_mesh(float *x, float *y, float *z) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(i < IMG_DIM && j < IMG_DIM)) return;

    unsigned idx = FLAT(i, j, IMG_DIM);
    x[idx] = j;
    y[idx] = IMG_DIM - i - 1;
    z[idx] = 0;
}

Mesh init_mesh(const unsigned hgt, const unsigned wid) {
    unsigned n_bytes = hgt * wid * sizeof(float);

    Mesh out;
    out.hgt = hgt;
    out.wid = wid;

    cudaMalloc((void **)&out.x, n_bytes);
    cudaMalloc((void **)&out.y, n_bytes);
    cudaMalloc((void **)&out.z, n_bytes);

    dim3 dimGrid(N_BLK(out.hgt, BLKSIZE_2D), N_BLK(out.wid, BLKSIZE_2D));
    dim3 dimBlk(BLKSIZE_2D, BLKSIZE_2D);
    dev_init_mesh<<<dimGrid, dimBlk>>>(out.x, out.y, out.z);
    cudaDeviceSynchronize();

    return out;
}