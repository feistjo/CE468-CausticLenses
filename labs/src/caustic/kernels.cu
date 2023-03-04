#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <cutil.h>

#include "util.h"
#include "kernels.h"

#define KERN_1D(nthreads) ((nthreads) + BLK_SIZE - 1) / BLK_SIZE, BLK_SIZE
