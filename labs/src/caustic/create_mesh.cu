#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cutil.h>

#include "util.h"
#include "kernels.h"


void optimize_mesh(cublasHandle_t ch, Matrix img, Mesh mesh, Matrix loss) {
    unsigned len = img.hgt * img.wid;

    compute_loss(mesh, img, loss);

    // normalize loss: loss = loss - (sum(loss) / len)
    float loss_sum = 0; 
    cublasSasum(ch, len, loss.elems, 1, &loss_sum);
    scalar_add(loss, -loss_sum / len, loss);

    // TODO: calculate phi and such
}

int create_mesh(Matrix host_img) {
    // CUBLAS stuff
    cublasHandle_t ch = NULL;
    cudaStream_t stream = NULL;
    cublasCreate(&ch);
    cublasSetStream(ch, stream);

    unsigned len = host_img.hgt * host_img.wid;

    Matrix img = to_device(host_img);

    // The mesh stores the 3D position of each point on the lens
    Mesh mesh = init_mesh(img.hgt + 1, img.wid + 1);
    
    // calculate sums
    float mesh_sum = float(len);
    float img_sum = 0;
    cublasSasum(ch, len, img.elems, 1, &img_sum);

    // boost brightness of image
    float boost_ratio = mesh_sum / img_sum;
    cublasSscal(ch, len, &boost_ratio, img.elems, 1);

    Matrix loss = init_matrix(host_img.hgt, host_img.wid);

    // optimize mesh until convergence
    // TODO: using only 1 iteration right now
    optimize_mesh(ch, img, mesh, loss);

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

    cudaFree(img.elems);
    cudaFree(loss.elems);
    return 0;
}