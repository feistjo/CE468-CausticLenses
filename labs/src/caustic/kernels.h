#ifndef KERNELS
#define KERNELS

#include "util.h"

void scalar_add(const Matrix a, const float b, Matrix out);

void compute_loss(Mesh mesh, Matrix img, Matrix loss);
float relax(Matrix m, Matrix loss, cublasHandle_t ch);
void march_mesh(Mesh mesh, Matrix phi);

#endif
