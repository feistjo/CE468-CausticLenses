#ifndef UTIL
#define UTIL

typedef struct {
    unsigned hgt;
    unsigned wid;
    float* elems;
} Matrix;

typedef struct {
    // a mesh is a 2D grid of hgt * wid elements
    unsigned hgt;
    unsigned wid;
    // each element in the grid has an x, y, and z coordinate
    float* x;
    float* y;
    float* z;
} Mesh;

typedef struct {
    float x;
    float y;
    float z;
} Point3D;

#define IMG_DIM 512
#define BLKSIZE_1D 256
#define BLKSIZE_2D 16

#define FLAT(row, col, n_cols) ((row) * (n_cols) + (col))
#define N_BLK(nthreads, bsize) (((nthreads) + bsize - 1) / bsize)
#define KERN_1D(nthreads) N_BLK(nthreads, B_SIZE_1D), B_SIZE_1D

Matrix init_matrix(const unsigned hgt, const unsigned wid);
Matrix to_device(const Matrix m);

Mesh init_mesh(const unsigned hgt, const unsigned wid);

#endif