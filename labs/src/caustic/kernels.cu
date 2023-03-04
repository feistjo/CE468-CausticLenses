#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <cutil.h>

#include "util.h"
#include "kernels.h"

inline __device__ float triangle_area(Point3D a, Point3D b, Point3D c) {
    return fabsf((a.x * (b.y - c.y)) + (b.x * (c.y - a.y) + c.x * (a.y - b.y) / 2));
}

// could be tiled and put into shared mem or something to improve performance
// right now we are doing multiple loads to global mem for each elem
// UNTESTED: hopefully it works
__global__ void dev_get_mesh_area(float *x, float *y, float *z, float *area) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(i < IMG_DIM && j < IMG_DIM)) return;
    
    unsigned tl = FLAT(i, j, IMG_DIM);
    unsigned tr = FLAT(i, j + 1, IMG_DIM);
    unsigned bl = FLAT(i + 1, j, IMG_DIM);
    unsigned br = FLAT(i + 1, j + 1, IMG_DIM);

    area[FLAT(i, j, IMG_DIM)] = triangle_area((Point3D){x[bl], y[bl], z[bl]}, (Point3D){x[tr], y[tr], z[tr]}, (Point3D){x[tl], y[tl], z[tl]})
                              + triangle_area((Point3D){x[bl], y[bl], z[bl]}, (Point3D){x[br], y[br], z[br]}, (Point3D){x[tr], y[tr], z[tr]});
}

void get_mesh_area(Mesh mesh, Matrix area) {
    dim3 dimGrid(N_BLK(mesh.hgt, BLKSIZE_2D), N_BLK(mesh.wid, BLKSIZE_2D));
    dim3 dimBlk(BLKSIZE_2D, BLKSIZE_2D);
    dev_get_mesh_area<<<dimGrid, dimBlk>>>(mesh.x, mesh.y, mesh.z, area.elems);
    cudaDeviceSynchronize();
}
