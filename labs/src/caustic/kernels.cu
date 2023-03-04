#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

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
