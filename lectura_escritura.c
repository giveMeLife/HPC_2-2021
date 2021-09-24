#include<stdio.h>
#include<emmintrin.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#define SIZE 65536

//abres la imagen y la guarda en buffer_out.
void read_image(char* file_name, int * buffer_out, int M, int N){
    int descriptor;
    size_t r;
    int size = N*M;
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
void raw_to_matrix(int* buffer_in, int** matrix, int M, int N){
    int k = 0;
    for(int i = 0; i<M; i++){
        for(int j = 0; j<N; j++){
            matrix[i][j] = buffer_in[k];
            k++;
        }
    }
}

//transforma una matriz en un arreglo unidimensional, el cual es buffer_out. Tiene que ser de tamano n elevado a 2, o size elevado a dos.
void matrix_to_raw(int* buffer_out, int** matrix, int size1, int size2){
    int k = 0;
    for(int i = 0; i<size1; i++){
        for(int j = 0; j<size2; j++){
            buffer_out[k] = matrix[i][j];
            k++;
        }
    }
}

//escribe la imagen. Se debe ingresar el arreglo raw, el cual es buffer_in.
void write_image(char* file_name, int* buffer_in, int size1, int size2){
    int descriptor;
    size_t r;
    int size = size1*size2;
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

void hough_algorithm(int ** matrix, int M, int N, int T, double dTeta, int ** H, double dR){
  for (int x = 0; x < M; x++){
    for (int y = 0; y < N; y++){
      if (matrix[x][y] != 0){
        for (int i = 0; i < T; i ++){
          double r = (x* cos((i)*dTeta) + y* sin((i)*dTeta));
          int r2 = (int)(r/dR);
          H[i][r2] =  H[i][r2] + 1;
        }
        
      }
    }
  }
}


int ** matrix_hough(int M, int R, int ** H){
  for (int i = 0; i < M; ++i){
    for (int j = 0; j < R; ++j){
      H[i][j] = 0;
  }
    }
    return H;

}

void umbral(int ** H, int M, int R, int U){
    for (int i = 0; i < M; ++i)
    {
       for (int j = 0; j < R; ++j)
       {
           if(H[i][j] > U){
               H[i][j] = 255;
           }
           else{
              H[i][j] = 0;

            }
       }
    }
}

void parallel_hough_algorithm(int ** matrix, int M, int N, int T, float dTeta, int ** H, float dR, float* angles){
  for (int x = 0; x < M; x++){
    for (int y = 0; y < N; y++){
      if (matrix[x][y] != 0){
        for (int i = 0; i < T/4*4; i += 4){
          __m128 dTeta2 = _mm_set1_ps(dTeta);
          __m128 dR2 = _mm_set1_ps(1/dR);

          __m128 x_position = _mm_set1_ps((float)x);
          __m128 y_position = _mm_set1_ps((float)y);

          __m128 indexes = _mm_load_ps(&angles[i]);
          __m128 angle = _mm_mul_ps(dTeta2, indexes);
          __m128 cosines = _mm_setr_ps(cos(angle[0]), cos(angle[1]), cos(angle[2]), cos(angle[3]));
          __m128 sines = _mm_setr_ps(sin(angle[0]), sin(angle[1]), sin(angle[2]), sin(angle[3]));
          
          cosines = _mm_mul_ps(cosines, x_position);
          sines = _mm_mul_ps(sines, y_position);

          __m128 radius = _mm_add_ps(cosines, sines);

          __m128 normalized_radius = _mm_mul_ps(radius, dR2);

          H[i][(int)normalized_radius[0]] =  H[i][(int)normalized_radius[0]] + 1;
          H[i+1][(int)normalized_radius[1]] =  H[i][(int)normalized_radius[1]] + 1;
          H[i+2][(int)normalized_radius[2]] =  H[i][(int)normalized_radius[2]] + 1;
          H[i+3][(int)normalized_radius[3]] =  H[i][(int)normalized_radius[3]] + 1;
        }
      }
    }
  }
}

int main(int argc, char *argv[]){
  int c;
  int N = 0;
  int T = 0;
  int R = 0;
  int U = 0;
  int M;
  char * inputImg = malloc(sizeof(char)*30);
  char * outputImg = malloc(sizeof(char)*30);

  while ((c = getopt(argc, argv, "i:o:M:N:T:R:U:")) != -1)
    switch (c)
      {
      case 'i':
        strcpy(inputImg, optarg);
        break;
      case 'o':
        strcpy(outputImg, optarg);
        break;
      case 'M':
        M = atof(optarg);
        break;
      case 'N':
        N = atof(optarg);
        break;
      case 'T':
        T = atof(optarg);
        break;
      case 'R':
        R = atof(optarg);
        break;
      case 'U':
        U = atof(optarg);
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
        return 1;
      default:
        abort ();
      }
    double theta = M_PI;
    double dTeta = (theta)/(T);
    double diagonal = sqrt(M*M + N*N);
    double dR = diagonal/(R);


    //char filename[] = "simplehough1-256x256.raw";
    int* buffer = (int*)malloc(sizeof(int)*M*N);

    read_image(inputImg, buffer, M,N);
    int** matrix = (int**)malloc(sizeof(int*)*M);
    for(int i = 0; i < M; i++){
        matrix[i] = (int*)malloc(sizeof(int)*N);
    }
    raw_to_matrix(buffer, matrix, M,N);

    int** H = (int**)malloc(sizeof(int*)*T);
    for(int i = 0; i < T; i++){
      H[i] = (int*)malloc(sizeof(int)*R);
    }
    H = matrix_hough(T,R,H);  
    //hough_algorithm(matrix,M, N, T,  dTeta,H, dR);

    float* angles = malloc(sizeof(float)*T);
    for(int i = 0; i < T; i++){
      angles[i] = (float)i;
    }

    parallel_hough_algorithm(matrix, M, N, T, (float)dTeta, H, (float)dR, angles);

    umbral(H,T,R,U);
    
    int* buffer2 = (int*)malloc(sizeof(int)*T*R);
    matrix_to_raw(buffer2, H, T, R);

    //char fileout[] = "b.raw";
    write_image(outputImg, buffer2, T,R);
    /*
    __m128d resultado;
    float arreglo[T];
    for (int i = 0; i < T; ++i){
      arreglo[i] = (float)i;
    }
   
    __m128 dTeta2 = _mm_set1_ps(dTeta);
    __m128 indice = _mm_load_ps(&arreglo[0]);
    __m128 angulo = _mm_mul_ps(dTeta2, indice);
    __m128 cosenos = _mm_setr_ps(cos(angulo[0]), cos(angulo[1]), cos(angulo[2]), cos(angulo[3]));
   // __m128 cos = _mm_cos_ps(angulo);
    printf("%lf %lf %lf %lf", cosenos[0], cosenos[1], cosenos[2], cosenos[3]);*/
    
}
