#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cutil.h>

#include "util.h"
#include "kernels.h"

int create_mesh(Matrix img) {
    //create square mesh the size of image
    Matrix meshy;

    long mesh_sum = img.wid * img.hgt;
    long image_sum = 0; //sum(image), parallelized?/tiled
    float boost_ratio = mesh_sum / image_sum;

    Matrix img_d;
    //call multiply kernel to multiply each pixel by boost_ratio

    /*
    //iterations of algorithm
    for (int i = 0; i < 4; i++) {
        Matrix D;
        //call kernel to get pixel areas, subtract img, and subtract sum / (512*512) (for 512x512 image)
        //can save loss images (quantifyLoss!)

        Matrix phi;
        //build phi
        for (int i = 0; i < 10000; i++) {
            float max_update;
            //allocate max_update on device
            //call kernel to relax

            if (max_update < 0.00001) {
                printf("Convergence reached at step %d with max_update %f\n", i, max_update);
                break;
            }
        }
        //call kernel to march mesh
    }

    float artifact_size = 0.1;
    float focal_length = 0.2;

    Matrix heights_d;
    // call kernel to find surface of mesh (heights)
    // call kernel to create bottom of mesh (solidify)
    // write output file obj
    */

    return 0;
}