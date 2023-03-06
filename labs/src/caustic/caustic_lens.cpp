#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cutil.h>

#include "pgma_io.hpp"
#include "create_mesh.h"
#include "util.h"

int main() {
    int *pixels;
    int denominator;
    int hgt;
    int wid;

    pgma_read("./samples/baboon.ascii.pgm", wid, hgt, denominator, &pixels);

    Matrix img;
    img.hgt = hgt;
    img.wid = wid;
    img.elems = (float *)malloc(hgt * wid * sizeof(float));

    for (int i = 0; i < hgt * wid; i++) {
        img.elems[i] = (float)pixels[i] / denominator;
        // std::cout << "matrix: (" << img.elems[i] << "), pgm: (" << pixels[i] << ")" << std::endl;
    }

    free(pixels);

    Mesh mesh = create_mesh(img);

    std::ofstream f;
    f.open("out.obj");
    save_obj(mesh, f);
    f.close();

    free_mesh_on_host(mesh);
}
