#include "functions.h"

__host__ void read_image(char* file_name, unsigned short int * buffer_out, int M, int N){
    FILE* image_raw = fopen(file_name, "rb");
    fread(buffer_out, sizeof(unsigned short int), M*N, image_raw);
    
}

__global__ void histgmem(unsigned short int* buffer, int* histogram, int image_length){
    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned short int buffer_id = buffer[id];
    if(id<image_length){
        atomicAdd(&histogram[buffer_id],1);
    }
}

__global__ void histsmem(unsigned short int* buffer, int* histogram, int image_length){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int j = threadIdx.x;
    __shared__ int temporal[256];
    if(j == 0){
        for(int l = 0; l<256; l++)
            temporal[l] = 0;
    }
    __syncthreads();
    if(id<image_length){
        atomicAdd(&temporal[buffer[id]],1);
    }
    __syncthreads();
    if(j == 0){
        for(int l = 0; l<256; l++)
            atomicAdd(&histogram[l],temporal[l]);
    }
    

}

__global__ void vecadd(float *a, float *b, float *c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}