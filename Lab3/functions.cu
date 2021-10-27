#include "functions.h"

/*
Descripción: Función del host que abre la imagen y la almacena en un arreglo
Entrada: Nombre del archivo que contiene la imagen, buffer en donde los valores serán 
         almacenados, ancho y largo de la imagen.
Proceso: con fopen se abre la imagen, se lee y se almacena en buffer_out.
Salida: Arreglo con los valores de la imagen.
*/
__host__ void read_image(char* file_name, unsigned short int * buffer_out, int M, int N){
    FILE* image_raw = fopen(file_name, "rb");
    fread(buffer_out, sizeof(unsigned short int), M*N, image_raw);
    
}

/*
Descripción: La función calcula el histograma de la imagen en memoria global.
Entrada: Buffer con los valores de la imagen, histograma en donde almacenarán los datos
         y dimensiones de la imagen. 
Proceso: Se calcula el id global de la hebra y mediante atomicAdd se suma 1 en cada posición
         del histograma cuando la imagen tenga ese valor. Se utiliza atomic debido a que asegura
         que no exista problema al acceder a memoria. 
         Un aspecto que se considera es que el valor que toma la hebra global no puede ser mayor al tamaño
         de la imagen.
Salida: Histograma con las frecuencias de los valores de la imagen
*/
__global__ void histgmem(unsigned short int* buffer, int* histogram, int image_length){
    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned short int buffer_id = buffer[id];
    if(id<image_length){
        atomicAdd(&histogram[buffer_id],1);
    }
}

/*
Descripción: La función calcula el histograma de la imagen en memoria compartida.
Entrada: Buffer con los valores de la imagen, histograma en donde almacenarán los datos
         y dimensiones de la imagen.
Proceso: Se calcula el id global de la hebra y local. Además se crea un arreglo de histogramas temporal
         para cada uno de los bloques. Luego, mediante atomicAdd se suma 1 en cada posición
         del histograma cuando la imagen tenga ese valor. A continuación se usa syncthreads para que todos
         los bloques terminen su ejecución y finalmente se utiliza otra vez atomiAdd, pero esta vez para 
         sumar todos los valores almacenados en los histogramas compartidos en un histograma global. 
         Un aspecto que se considera es que el valor que toma la hebra global no puede ser mayor al tamaño
         de la imagen.
Salida: Histograma con las frecuencias de los valores de la imagen
*/
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

