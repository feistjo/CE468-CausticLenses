#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

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
    y[idx] = i;
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

    dim3 dimGrid(N_BLK(hgt, BLKSIZE_2D), N_BLK(wid, BLKSIZE_2D));
    dim3 dimBlk(BLKSIZE_2D, BLKSIZE_2D);
    dev_init_mesh<<<dimGrid, dimBlk>>>(out.x, out.y, out.z);
    cudaDeviceSynchronize();

    return out;
}

void free_mesh(Mesh mesh) {
    cudaFree(mesh.x);
    cudaFree(mesh.y);
    cudaFree(mesh.z);
}

void save_obj(Mesh mesh, std::ostream &f, float scale, float scalez) {
    for (int i = 0; i < mesh.hgt * mesh.wid; i++)
    {
        f << "v " << mesh.x[i] * scale << " " << mesh.y[i] * scale << " " << mesh.z[i] * scalez << '\n';
    }

    for (int i = 0; i < mesh.hgt; i++)
    {
        for (int j = 0; j < mesh.wid; j++)
        {
            unsigned tri[2][3];
    
            tri[0][0] = FLAT(i, j, IMG_DIM);
            tri[0][1] = FLAT(i, j + 1, IMG_DIM);
            tri[0][2] = FLAT(i + 1, j, IMG_DIM);

            tri[1][0] = FLAT(i + 1, j + 1, IMG_DIM);
            tri[1][1] = FLAT(i + 1, j, IMG_DIM);
            tri[1][2] = FLAT(i, j + 1, IMG_DIM);

            f << "f " << tri[0][0] << " " << tri[0][1] << " " << tri[0][2] << '\n';
            f << "f " << tri[1][0] << " " << tri[1][1] << " " << tri[1][2] << '\n';
        }

        f << "dims " << mesh.wid << " " << mesh.hgt << '\n';
    }
}