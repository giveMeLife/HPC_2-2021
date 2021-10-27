#include "functions.h"



__host__ int main(int argc, char *argv[]){

    int c;
    int m;
    int n;
    int t;
    int d;
    char* i = (char*) malloc(sizeof(char)*30);
    char* o = (char*) malloc(sizeof(char)*30);
    while ((c = getopt(argc, argv, "i:m:n:o:t:d:")) != -1)
        switch (c)
        {
        case 'i':
            strcpy(i, optarg);
            break;
        case 'm':
            m = atoi(optarg);
        case 'n':
            n = atoi(optarg);
        case 'o':
            strcpy(o, optarg);
            break;
        case 't':
            t = atoi(optarg);
            break;
        case 'd':
            d = atoi(optarg);
            break;
        case '?':
            if (optopt == 'c')
            fprintf (stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint (optopt))
            fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
            fprintf (stderr,
                    "Unknown option character `\\x%x'.\n",
                    optopt);
        }


    int image_length = m*n;
    unsigned short int* buffer = (unsigned short int*)malloc(sizeof(unsigned short int)*image_length);
    int* histogram = (int*)malloc(sizeof(int)*256);
    for(int j = 0; j<256; j++){
        histogram[j] = 0;
    }
    read_image(i , buffer, m, n);
    /*for(int i = 0; i<512; i++){
        printf("%hu\n", buffer[i]);
    }*/

    int* device_histogram;
    int* device_histogram2;

    unsigned short int* device_buffer;
    
    cudaMalloc((void**) &device_histogram, 256*sizeof(int));
    cudaMalloc((void**) &device_buffer, image_length*sizeof(unsigned short int));

    cudaMemcpy(device_histogram, histogram, 256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_buffer, buffer, image_length*sizeof(unsigned short int), cudaMemcpyHostToDevice);
    
    histgmem<<<ceil((image_length)/t),t>>>(device_buffer, device_histogram, image_length);
    
    int* hist_final = (int*)malloc(sizeof(int)*256);
    cudaMemcpy(hist_final, device_histogram, 256*sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int j = 0; j<256; j++){
        printf("%d - %d\n",j, hist_final[j]);
    }
    printf("**************************\n");
    
    cudaMalloc((void**) &device_histogram2, 256*sizeof(int));

    histsmem<<<ceil((image_length)/t),t>>>(device_buffer, device_histogram2, image_length);

    int* hist_final2 = (int*)malloc(sizeof(int)*256);
    cudaMemcpy(hist_final2, device_histogram2, 256*sizeof(int), cudaMemcpyDeviceToHost);
     for(int j = 0; j<256; j++){
        printf("%d - %d\n",j, hist_final2[j]);
    }
    printf("%d",d);
/*
    int N = 1024;
    float *a = (float *) malloc(N*sizeof(float));
    float *b = (float *) malloc(N*sizeof(float));
    float *c = (float *) malloc(N*sizeof(float));
    int i;
    for (i=0; i < N; i++) {
        a[i] = 1;
        b[i] = 2;
    }
    // allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, N*sizeof(float));
    cudaMalloc((void **) &d_b, N*sizeof(float));
    cudaMalloc((void **) &d_c, N*sizeof(float));
    
    // copy host memory to device memory
    cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
    vecadd<<<N/512, 512>>>(d_a, d_b, d_c);
    // copy result from device memory to host
    cudaMemcpy(c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
    for (i=0; i < N; i++)
        printf("%f\n", c[i]);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
*/
    return 0;

}