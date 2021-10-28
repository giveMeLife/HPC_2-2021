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
    /*  *****************************************************************
        *****************************************************************
        ***********************Memoria global****************************
        *****************************************************************
        ***************************************************************** */

    //Se lee la imagen y se almacena en buffer. Además se inicializa el histograma en 0
    int image_length = m*n;
    unsigned short int* buffer = (unsigned short int*)malloc(sizeof(unsigned short int)*image_length);
    int* histogram = (int*)malloc(sizeof(int)*256);
    for(int j = 0; j<256; j++){
        histogram[j] = 0;
    }
    read_image(i , buffer, m, n);

    // Se asigna memoria en device para el histograma y la imagen
    int* device_histogram;
    int* hist_final = (int*)malloc(sizeof(int)*256);
    unsigned short int* device_buffer;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMalloc((void**) &device_histogram, 256*sizeof(int));
    cudaMalloc((void**) &device_buffer, image_length*sizeof(unsigned short int));

    //Se copia el histograma y la imagen de host a device
    cudaMemcpy(device_histogram, histogram, 256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_buffer, buffer, image_length*sizeof(unsigned short int), cudaMemcpyHostToDevice);
    
    //Kernel memoria global
    histgmem<<<ceil((image_length)/t),t>>>(device_buffer, device_histogram, image_length);
    
    //Se copia el histograma de device a host
    cudaMemcpy(hist_final, device_histogram, 256*sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeinmilliseconds = 0;
    cudaEventElapsedTime(&timeinmilliseconds, start, stop);

    printf("Tiempo global: %f\n", timeinmilliseconds);
    /* *****************************************************************
       *****************************************************************
       *************************Memoria compartida**********************
       *****************************************************************
       ***************************************************************** */
       

    int* device_histogram2;   
    int* hist_final2 = (int*)malloc(sizeof(int)*256);


    cudaEventRecord(start);
    // Se asigna memoria en device para el histograma
    cudaMalloc((void**) &device_histogram2, 256*sizeof(int));
    cudaMalloc((void**) &device_buffer, image_length*sizeof(unsigned short int));

    cudaMemcpy(device_buffer, buffer, image_length*sizeof(unsigned short int), cudaMemcpyHostToDevice); 
    cudaMemcpy(device_histogram2, histogram, 256*sizeof(int), cudaMemcpyHostToDevice);

    //Kernel memoria compartida
    histsmem<<<ceil((image_length)/t),t>>>(device_buffer, device_histogram2, image_length);

    //Se copia el histograma de device a host
    cudaMemcpy(hist_final2, device_histogram2, 256*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    timeinmilliseconds = 0;
    cudaEventElapsedTime(&timeinmilliseconds, start, stop);
    printf("Tiempo shared: %f\n", timeinmilliseconds);

    if(d==1){
        debug(hist_final, hist_final2);
    }

    write_histogram(o, hist_final, hist_final2);

    return 0;

}