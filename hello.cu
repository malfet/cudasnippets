#include <stdio.h>
__global__ void kernel() {
  printf("Hello World of CUDA %d\n", threadIdx.x);
}

int main() {
   kernel<<<1,1>>>();
  return cudaDeviceSynchronize();
}
