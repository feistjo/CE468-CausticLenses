#ifndef UTIL
#define UTIL

typedef struct {
    unsigned int wid;
    unsigned int hgt;
    float* elems;
} Matrix;

#define FLATTEN(row, col, n_cols) ((row) * (n_cols) + (col))

Matrix to_device(const Matrix m);

#endif