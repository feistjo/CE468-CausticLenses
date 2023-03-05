#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <cublas_v2.h>
#include <cutil.h>

#include "util.h"
#include "kernels.h"

__global__ void dev_scalar_add(const float *a, const float b, float *out) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    if (!(i < IMG_DIM * IMG_DIM)) return;

    out[i] = a[i] + b;
}

void scalar_add(const Matrix a, const float b, Matrix out) {
    dev_scalar_add<<<KERN_1D(out.hgt * out.wid)>>>(a.elems, b, out.elems);
    cudaDeviceSynchronize();
}

inline __device__ float triangle_area(Point3D a, Point3D b, Point3D c) {
    return fabsf((a.x * (b.y - c.y)) + (b.x * (c.y - a.y) + c.x * (a.y - b.y) / 2));
}

// could be tiled and put into shared mem or something to improve performance
// right now we are doing multiple loads to global mem for each elem
// UNTESTED: hopefully it works lol
__global__ void dev_compute_loss(float *x, float *y, float *z, float *img, float *loss) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(i < IMG_DIM && j < IMG_DIM)) return;

    unsigned idx = FLAT(i, j, IMG_DIM);
    
    unsigned tl = idx;
    unsigned tr = FLAT(i, j + 1, IMG_DIM);
    unsigned bl = FLAT(i + 1, j, IMG_DIM);
    unsigned br = FLAT(i + 1, j + 1, IMG_DIM);

    float cell_area = triangle_area({x[bl], y[bl], z[bl]}, {x[tr], y[tr], z[tr]}, {x[tl], y[tl], z[tl]})
                    + triangle_area({x[bl], y[bl], z[bl]}, {x[br], y[br], z[br]}, {x[tr], y[tr], z[tr]});

    loss[idx] = cell_area - img[idx];
}

void compute_loss(Mesh mesh, Matrix img, Matrix loss) {
    dim3 dimGrid(N_BLK(mesh.hgt, BLKSIZE_2D), N_BLK(mesh.wid, BLKSIZE_2D));
    dim3 dimBlk(BLKSIZE_2D, BLKSIZE_2D);
    dev_compute_loss<<<dimGrid, dimBlk>>>(mesh.x, mesh.y, mesh.z, img.elems, loss.elems);
    cudaDeviceSynchronize();
}

__global__ void dev_relax(float *a, float *loss, float *deltas) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(i < IMG_DIM && j < IMG_DIM)) return;

    unsigned idx = FLAT(i, j, IMG_DIM);

    // omega = 2 / (1 + pi / IMG_DIM)
    float OMEGA = 1.99;
    float count = 0;
    float sum = 0;

    // I tried desperately to fix the gross if statements with padding but I don't think there's a way to avoid them
    if (i == 0 && j == 0) {
        count = 2;
        sum = a[FLAT(i, j + 1, IMG_DIM)] + a[FLAT(i + 1, j, IMG_DIM)];
    } else if (i == 0 && j == IMG_DIM - 1) {
        count = 2;
        sum = a[FLAT(i, j - 1, IMG_DIM)] + a[FLAT(i + 1, j, IMG_DIM)];
    } else if (i == IMG_DIM - 1 && j == 0) {
        count = 2;
        sum = a[FLAT(i - 1, j, IMG_DIM)] + a[FLAT(i, j + 1, IMG_DIM)];
    } else if (i == IMG_DIM - 1 && j == IMG_DIM - 1) {
        count = 2;
        sum = a[FLAT(i - 1, j, IMG_DIM)] + a[FLAT(i, j - 1, IMG_DIM)];
    } else if (i == 0) {
        count = 3;
        sum = a[FLAT(i, j - 1, IMG_DIM)] + a[FLAT(i, j + 1, IMG_DIM)] + a[FLAT(i + 1, j, IMG_DIM)];
    } else if (i == IMG_DIM - 1) {
        count = 3;
        sum = a[FLAT(i, j - 1, IMG_DIM)] + a[FLAT(i, j + 1, IMG_DIM)] + a[FLAT(i - 1, j, IMG_DIM)];
    } else if (j == 0) {
        count = 3;
        sum = a[FLAT(i - 1, j, IMG_DIM)] + a[FLAT(i + 1, j, IMG_DIM)] + a[FLAT(i, j + 1, IMG_DIM)];
    } else if (j == IMG_DIM - 1) {
        count = 3;
        sum = a[FLAT(i - 1, j, IMG_DIM)] + a[FLAT(i + 1, j, IMG_DIM)] + a[FLAT(i, j - 1, IMG_DIM)];
    } else {
        count = 4;
        sum = a[FLAT(i - 1, j, IMG_DIM)] + a[FLAT(i + 1, j, IMG_DIM)] + a[FLAT(i, j - 1, IMG_DIM)] + a[FLAT(i, j + 1, IMG_DIM)];
    }

    float delta = OMEGA / count * (sum - (count * a[idx]) - loss[idx]);
    deltas[idx] = delta;
    a[idx] += delta;
}

float relax(Matrix m, Matrix loss, cublasHandle_t ch) {
    float *deltas;
    cudaMalloc(&deltas, m.hgt * m.wid * sizeof(float));

    dim3 dimGrid(N_BLK(m.hgt, BLKSIZE_2D), N_BLK(m.wid, BLKSIZE_2D));
    dim3 dimBlk(BLKSIZE_2D, BLKSIZE_2D);
    dev_relax<<<dimGrid, dimBlk>>>(m.elems, loss.elems, deltas);
    cudaDeviceSynchronize();

    int idx_of_max;
    float max_update;
    cublasIsamax(ch, m.hgt * m.wid, deltas, 1, &idx_of_max);
    cudaMemcpy(&max_update, deltas + idx_of_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deltas);
    return max_update;
}