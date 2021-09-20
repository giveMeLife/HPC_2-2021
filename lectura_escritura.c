#include<stdio.h>
#include<emmintrin.h>
#include <unistd.h>
#include <fcntl.h>

#define SIZE 65536

//abres la imagen y la guarda en buffer_out.
void read_image(char* file_name, int * buffer_out, int size){
    int descriptor;
    size_t r;
    size = size*size;
    descriptor = open(file_name, O_RDONLY);
    if( descriptor == -1 ){
        fprintf(stderr,"Error abriendo imagen: %s\n",file_name);
        exit(1);
    }
    else{
        fprintf(stderr,"Imagen %s abierta correctamente\n",file_name);
    }
    r = read(descriptor, buffer_out, size*sizeof(int));
    close(descriptor);


}

//transforma el arreglo (ya que la imagen al final es un arreglo de una dimension) a una matrix x,y de tamano size,size.
void raw_to_matrix(int* buffer_in, int** matrix, int size){
    int k = 0;
    for(int i = 0; i<size; i++){
        for(int j = 0; j<size; j++){
            matrix[i][j] = buffer_in[k];
            k++;
        }
    }
}

//transforma una matriz en un arreglo unidimensional, el cual es buffer_out. Tiene que ser de tamano n elevado a 2, o size elevado a dos.
void matrix_to_raw(int* buffer_out, int** matrix, int size){
    int k = 0;
    for(int i = 0; i<size; i++){
        for(int j = 0; j<size; j++){
            buffer_out[k] = matrix[i][j];
            k++;
        }
    }
}

//escribe la imagen. Se debe ingresar el arreglo raw, el cual es buffer_in.
void write_image(char* file_name, int* buffer_in, int size){
    int descriptor;
    size_t r;
    size = size*size;
    descriptor = open(file_name, O_WRONLY | O_CREAT);
    if( descriptor == -1 ){
        fprintf(stderr,"Error creando imagen: %s\n",file_name);
        exit(1);
    }
    else{
        fprintf(stderr,"Imagen %s creada correctamente\n",file_name);
    }
    r = write(descriptor, buffer_in, size*sizeof(int));
    close(descriptor);


}

int main(void){
    char filename[] = "simplehough1-256x256.raw";
    int n = 256;
    int* buffer = (int*)malloc(sizeof(int)*n*n);

    read_image(filename, buffer, n);
    int** matrix = (int**)malloc(sizeof(int*)*n);
    for(int i = 0; i < n; i++){
        matrix[i] = (int*)malloc(sizeof(int)*n);
    }
    raw_to_matrix(buffer, matrix, n);

    int* buffer2 = (int*)malloc(sizeof(int)*n*n);
    matrix_to_raw(buffer2, matrix, n);

    char fileout[] = "testfile.raw";
    write_image(fileout, buffer2, n);
    
}
