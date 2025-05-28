#include <stdio.h>
#include "geometry.h"
#include "firelib_cuda.h"

__global__ void cuda_hello(){
      printf("Hello World from GPU!\n");
}
