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
#define MESH_DIM (IMG_DIM + 1)
#define BLKSIZE_1D 256
#define BLKSIZE_2D 16

#define FLAT(row, col, n_cols) ((row) * (n_cols) + (col))
#define N_BLK(nthreads, bsize) (((nthreads) + bsize - 1) / bsize)
#define KERN_1D(nthreads) N_BLK(nthreads, BLKSIZE_1D), BLKSIZE_1D

Matrix init_matrix(const unsigned hgt, const unsigned wid);
Matrix to_device(const Matrix m);

Mesh init_mesh_on_dev(const unsigned hgt, const unsigned wid);
Mesh to_host(Mesh mesh_d);
void free_mesh_on_device(Mesh mesh);
void free_mesh_on_host(Mesh mesh);

void save_obj(Mesh mesh, std::ostream &f, float scale = 1.0f, float scalez = 1.0f);

#endif