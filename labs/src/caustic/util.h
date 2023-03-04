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

#define IMG_DIM 512
#define BLK_SIZE 512
#define FLATTEN(row, col, n_cols) ((row) * (n_cols) + (col))

Matrix to_device(const Matrix m);

#endif