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

    if (!(i < MESH_DIM - 1 && j < MESH_DIM - 1)) return;

    unsigned idx = FLAT(i, j, IMG_DIM);
    
    unsigned tl = FLAT(i, j, MESH_DIM);
    unsigned tr = FLAT(i, j + 1, MESH_DIM);
    unsigned bl = FLAT(i + 1, j, MESH_DIM);
    unsigned br = FLAT(i + 1, j + 1, MESH_DIM);

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

__device__ float atomicMax(float* address, float val)
{
    float old = *address, assumed;

    do {
        assumed = old;
        if (val > assumed)
        {
            old = atomicCAS((uint32_t*)address, *(uint32_t*)&assumed, *(uint32_t*)&val);
        }

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (*(int*)&assumed != *(int*)&old);

    return old;
}

__global__ void dev_relax(float *a, float *loss, float *max_delta, bool even_odd) {
    unsigned i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned j = (blockIdx.y * blockDim.y) + threadIdx.y;
    i *= 2;
    if (j%2 != even_odd) i++;
    
    if (!(i < IMG_DIM && j < IMG_DIM))
    {
        return;
    }

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
    a[idx] += delta;
    // if (idx == 0) printf("phi: %f\n", a[idx]);

    __syncthreads();

    atomicMax(max_delta, delta);
    __syncthreads();
}

float relax(Matrix m, Matrix loss) {
    float *max_delta;
    float max_delta_h = 0;
    cudaMalloc(&max_delta, sizeof(float));
    cudaMemcpy(max_delta, &max_delta_h, sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(N_BLK(m.hgt, BLKSIZE_2D) / 2, N_BLK(m.wid, BLKSIZE_2D));
    dim3 dimBlk(BLKSIZE_2D, BLKSIZE_2D);
    dev_relax<<<dimGrid, dimBlk>>>(m.elems, loss.elems, max_delta, 0);
    cudaDeviceSynchronize();
        cudaMemcpy(&max_delta_h, max_delta, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Max delta 1: %f\n", max_delta_h);
    dev_relax<<<dimGrid, dimBlk>>>(m.elems, loss.elems, max_delta, 1);
    cudaDeviceSynchronize();
    //printf("Relaxed\n");

    cudaMemcpy(&max_delta_h, max_delta, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(max_delta);
    return max_delta_h;
}

inline __device__ float grad_i(unsigned i, unsigned j, float *phi) {
    return i == IMG_DIM - 1 ? 0 : phi[FLAT(i + 1, j, IMG_DIM)] - phi[FLAT(i, j, IMG_DIM)];
}

inline __device__ float grad_j(unsigned i, unsigned j, float *phi) {
    return j == IMG_DIM - 1 ? 0 : phi[FLAT(i, j + 1, IMG_DIM)] - phi[FLAT(i, j, IMG_DIM)];
}

inline __device__ void find_t(Point3D p1, Point3D p2, Point3D p3, Point3D dp1, Point3D dp2, Point3D dp3, float* t1, float* t2) {
    float x1 = p2.x - p1.x;
    float y1 = p2.y - p1.y;

    float x2 = p3.x - p1.x;
    float y2 = p3.y - p1.y;

    float u1 = dp2.x - dp1.x;
    float v1 = dp2.y - dp1.y;

    float u2 = dp3.x - dp1.x;
    float v2 = dp3.y - dp1.y;

    float a = u1 * v2 - u2 * v1;
    float b = x1 * v1 + y2 * u1 - x2 * v1 - y1 * u2;
    float c = x1 * y2 - x2 * y1;

    if (a != 0)
    {
        float quotient = (b*b) - (4*a*c);
        if (quotient >= 0)
        {
            float d = sqrtf(quotient);
            *t1 = (-b - d) / (2*a);
            *t2 = (-b + d) / (2*a);
            return;
        }
        else
        {
            *t1 = -123.0f;
            *t2 = -123.0f;
            return;
        }
    }
    else
    {
        *t1 = -c / b;
        *t2 = -c / b;
        return;
    }
}

__device__ float atomicMin(float* address, float val)
{
    float old = *address, assumed;

    do {
        assumed = old;
        if (val < assumed)
        {
            old = atomicCAS((uint32_t*)address, *(uint32_t*)&assumed, *(uint32_t*)&val);
        }

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (*(int*)&assumed != *(int*)&old);

    return old;
}

__global__ void dev_prepare_march_mesh(float *x, float *y, float *z, float *phi, float *vel_x, float *vel_y, float *min_t) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(i < MESH_DIM && j < MESH_DIM)) return;

    unsigned idx = FLAT(i, j, MESH_DIM);

    float vel_i, vel_j;

    if (i == MESH_DIM - 1) {
        vel_i = 0;
    } else if (j == MESH_DIM - 1) {
        vel_i = grad_i(i, j - 1, phi);
    } else {
        vel_i = grad_i(i, j, phi);
    }

    if (j == MESH_DIM - 1) {
        vel_j = 0;
    } else if (i == MESH_DIM - 1) {
        vel_j = grad_j(i - 1, j, phi);
    } else {
        vel_j = grad_j(i, j, phi);
    }

    vel_x[idx] = -vel_j;
    vel_y[idx] = -vel_i;
    
    __syncthreads();

    if (!(i < MESH_DIM - 1 && j < MESH_DIM - 1)) return;
    
    // mesh triangles
    unsigned tri[2][3];
    
    tri[0][0] = FLAT(i, j, MESH_DIM);
    tri[0][1] = FLAT(i, j + 1, MESH_DIM);
    tri[0][2] = FLAT(i + 1, j, MESH_DIM);
    
    tri[1][0] = FLAT(i + 1, j + 1, MESH_DIM);
    tri[1][1] = FLAT(i + 1, j, MESH_DIM);
    tri[1][2] = FLAT(i, j + 1, MESH_DIM);

    float t1 = 0;
    float t2 = 0;
    float t3 = 0;
    float t4 = 0;

    find_t({x[tri[0][0]],     y[tri[0][0]],     z[tri[0][0]]},
           {x[tri[0][1]],     y[tri[0][1]],     z[tri[0][1]]},
           {x[tri[0][2]],     y[tri[0][2]],     z[tri[0][2]]},
           {vel_x[tri[0][0]], vel_y[tri[0][0]], 0           },
           {vel_x[tri[0][1]], vel_y[tri[0][1]], 0           },
           {vel_x[tri[0][2]], vel_y[tri[0][2]], 0           },
           &t1, &t2);
    
    find_t({x[tri[1][0]],     y[tri[1][0]],     z[tri[1][0]]},
           {x[tri[1][1]],     y[tri[1][1]],     z[tri[1][1]]},
           {x[tri[1][2]],     y[tri[1][2]],     z[tri[1][2]]},
           {vel_x[tri[1][0]], vel_y[tri[1][0]], 0           },
           {vel_x[tri[1][1]], vel_y[tri[1][1]], 0           },
           {vel_x[tri[1][2]], vel_y[tri[1][2]], 0           },
           &t3, &t4);

    __syncthreads();

    float local_min_t = t1 < t2 ? t1 : t2;
    local_min_t = local_min_t < t3 ? local_min_t : t3;
    local_min_t = local_min_t < t4 ? local_min_t : t4;

    __syncthreads();

    if (local_min_t >= 0) atomicMin(min_t, local_min_t);
}

__global__ void dev_march_mesh(float *x, float *y, float *vel_x, float *vel_y, float *min_t) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(i < MESH_DIM && j < MESH_DIM)) return;

    unsigned idx = FLAT(i, j, MESH_DIM);

    float delta = *min_t / 2;
    x[idx] += vel_x[idx] * delta;
    y[idx] += vel_y[idx] * delta;
}

void march_mesh(Mesh mesh, Matrix phi) {
    unsigned n_bytes = mesh.hgt * mesh.wid * sizeof(float);
    float *vel_x;
    float *vel_y;
    float *min_t;

    cudaMalloc(&vel_x, n_bytes);
    cudaMalloc(&vel_y, n_bytes);
    cudaMalloc(&min_t, sizeof(float));
    float min_t_h = 10000;
    cudaMemcpy(min_t, &min_t_h, sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(N_BLK(mesh.hgt, BLKSIZE_2D), N_BLK(mesh.wid, BLKSIZE_2D));
    dim3 dimBlk(BLKSIZE_2D, BLKSIZE_2D);
    dev_prepare_march_mesh<<<dimGrid, dimBlk>>>(mesh.x, mesh.y, mesh.z, phi.elems, vel_x, vel_y, min_t);
    cudaDeviceSynchronize();
    dev_march_mesh<<<dimGrid, dimBlk>>>(mesh.x, mesh.y, vel_x, vel_y, min_t);
    cudaDeviceSynchronize();

    cudaFree(vel_x);
    cudaFree(vel_y);
    cudaFree(min_t);
}