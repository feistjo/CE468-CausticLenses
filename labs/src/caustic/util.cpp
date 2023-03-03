#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cutil.h>

#include "util.h"

Matrix to_device(const Matrix m) {
    Matrix m_d = m;
    cudaMalloc((void**)&m_d.elems, m.wid * m.hgt * sizeof(float));
    return m_d;
}