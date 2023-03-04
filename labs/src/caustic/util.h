#ifndef UTIL
#define UTIL

typedef struct {
    unsigned int wid;
    unsigned int hgt;
    float* elems;
} Matrix;

#define IMG_DIM 512
#define BLK_SIZE 512
#define FLATTEN(row, col, n_cols) ((row) * (n_cols) + (col))

Matrix to_device(const Matrix m);

#endif