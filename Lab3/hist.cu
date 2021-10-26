#include "functions.h"



__host__ int main(){
/*
    int m;
    int n;
    int t;
    int d;
    char * i = malloc(sizeof(char)*30);
    char * o = malloc(sizeof(char)*30);

    int a;
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
*/
    int m = 1151;
    int n = 976;
    unsigned short int* buffer = (unsigned short int*)malloc(sizeof(unsigned short int)*m*n);
    int* histogram = (int*)malloc(sizeof(int)*256);
    for(int i = 0; i<256; i++){
        histogram[i] = 0;
    }
    char name[50] = "img1lab3-1151x976.raw";
    read_image(name , buffer, m, n);
    /*for(int i = 0; i<512; i++){
        printf("%hu\n", buffer[i]);
    }*/

    int* device_histogram;
    unsigned short int* device_buffer;
    int size = m*n;
    cudaMalloc((void**) &device_histogram, 256*sizeof(int));
    cudaMalloc((void**) &device_buffer, m*n*sizeof(unsigned short int));

    cudaMemcpy(device_histogram, histogram, 256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_buffer, buffer, m*n*sizeof(unsigned short int), cudaMemcpyHostToDevice);
    
    histgmem<<<size,1>>>(device_buffer, device_histogram);

    int* hist_final = (int*)malloc(sizeof(int)*256);
    cudaMemcpy(hist_final, device_histogram, 256*sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i<256; i++){
        printf("%d - %d\n",i, hist_final[i]);
    }

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