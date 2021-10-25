#include "functions.h"

__host__ void read_image(char* file_name, unsigned short int * buffer_out, int M, int N){
    FILE* image_raw = fopen(file_name, "rb");
    fread(buffer_out, sizeof(unsigned short int), M*N, image_raw);
    
}

__global__ void histgmem(unsigned short int* buffer, int* histogram){
    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned short int buffer_id = buffer[id];
    atomicAdd(&histogram[buffer_id],1);
}

__global__ void vecadd(float *a, float *b, float *c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}