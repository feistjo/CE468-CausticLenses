#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cutil.h>

#include "util.h"

Matrix to_device(const Matrix m) {
    Matrix m_d = m;
    size_t len = m.wid * m.hgt * sizeof(float);
    cudaMalloc((void **)&m_d.elems, len);
    cudaMemcpy(m_d.elems, m.elems, len, cudaMemcpyHostToDevice);
    return m_d;
}