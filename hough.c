#include <stdio.h>
#include <emmintrin.h>
#include <unistd.h>
#include <time.h>
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
    int size = N*M;
    descriptor = open(file_name, O_RDONLY);
    if( descriptor == -1 ){
        fprintf(stderr,"Error abriendo imagen: %s\n",file_name);
        exit(1);
    }
    else{
        fprintf(stderr,"Imagen %s abierta correctamente\n",file_name);
    }
    read(descriptor, buffer_out, size*sizeof(int));
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
    int size = size1*size2;
    descriptor = open(file_name, O_RDWR | O_CREAT);
    if( descriptor == -1 ){
        fprintf(stderr,"Error creando imagen: %s\n",file_name);
        exit(1);
    }
    else{
        fprintf(stderr,"Imagen %s creada correctamente\n",file_name);
    }
    write(descriptor, buffer_in, size*sizeof(int));
    close(descriptor);


}

void hough_vote(int ** hough_matrix, int i, int r, double dR, int R){
  int normalized_r;
  if(r < 0){
    normalized_r = (int)abs(((r/dR))+(double)R/2);
  }
  else{
    normalized_r = (int)((r/dR)+(double)R/2);
  }
  hough_matrix[i][normalized_r] =  hough_matrix[i][normalized_r] + 1;
}


void hough_algorithm(int ** matrix, int M, int N, int T, double dTeta, int ** H, double dR, double diagonal, double R){
  for (int x = 0; x < M; x++){
    for (int y = 0; y < N; y++){
      if (matrix[x][y] != 0){
        for (int i = 0; i < T; i ++){
          double r = (x* cos((i)*dTeta) + y* sin((i)*dTeta));
          hough_vote(H, i, r, dR, R);
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

void parallel_hough_algorithm(int ** matrix, int M, int N, int T, float dTeta, int ** H, float dR, float* angles, float diagonal, float R){
   __m128 dTeta2 = _mm_set1_ps(dTeta);
   __m128 dR2 = _mm_set1_ps(1/dR);
   float normalized_radius_att[4] __attribute__((aligned(16))) = { 0.0, 0.0, 0.0, 0.0 };
  for (int x = 0; x < M; x++){
     __m128 x_position = _mm_set1_ps((float)x);
    for (int y = 0; y < N; y++){
       __m128 y_position = _mm_set1_ps((float)y);
      if (matrix[x][y] != 0){
        for (int i = 0; i < T/4*4; i += 4){
          __m128 indexes = _mm_load_ps(&angles[i]);
          __m128 angle = _mm_mul_ps(dTeta2, indexes);
          __m128 cosines = _mm_setr_ps(cos(angle[0]), cos(angle[1]), cos(angle[2]), cos(angle[3]));
          __m128 sines = _mm_setr_ps(sin(angle[0]), sin(angle[1]), sin(angle[2]), sin(angle[3]));
          cosines = _mm_mul_ps(cosines, x_position);
          sines = _mm_mul_ps(sines, y_position);
          __m128 radius = _mm_add_ps(cosines, sines);
          __m128 normalized_radius = _mm_mul_ps(radius, dR2);
          _mm_store_ps(normalized_radius_att, normalized_radius); 
          hough_vote(H, i, normalized_radius_att[0], dR, R);
          hough_vote(H, i, normalized_radius_att[1], dR, R);
          hough_vote(H, i, normalized_radius_att[2], dR, R);
          hough_vote(H, i, normalized_radius_att[3], dR, R);
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
  clock_t start_t, end_t;
  double total_time;
  int a;
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
    double dR = 2*diagonal/(R);


    //Lectura del archivo y traspaso a matriz
    int* buffer = (int*)malloc(sizeof(int)*M*N);
    read_image(inputImg, buffer, M,N);
    //Matriz que almacena al archivo
    int** matrix = (int**)malloc(sizeof(int*)*M);
    for(int i = 0; i < M; i++){
        matrix[i] = (int*)malloc(sizeof(int)*N);
    }
    raw_to_matrix(buffer, matrix, M,N);

    //Creación de matriz H
    int** H = (int**)malloc(sizeof(int*)*T);
    for(int i = 0; i < T; i++){
      H[i] = (int*)malloc(sizeof(int)*R);
    }
    H = matrix_hough(T,R,H);  

    //Ejecución algoritmo secuencial
    start_t = clock();
    hough_algorithm(matrix,M, N, T, dTeta,H, dR, diagonal, R);
    end_t = clock();
    total_time = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("Tiempo de latencia parte secuencial: %lf\n", total_time  );

    //Arreglo de ángulos
    float* angles = malloc(sizeof(float)*T);
    for(int i = 0; i < T; i++){
      angles[i] = (float)i;
    }

    //Ejecución algoritmo paralelo
    start_t = clock();
    parallel_hough_algorithm(matrix, M, N, T, (float)dTeta, H, (float)dR, angles, diagonal, R);
    end_t = clock();
    total_time = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("Tiempo de latencia parte paralela: %lf\n", total_time  );

    //Umbralización de la solución
    umbral(H,T,R,U);
    
    //Escritura del resultado secuencial
    int* buffer2 = (int*)malloc(sizeof(int)*T*R);
    matrix_to_raw(buffer2, H, T, R);
    write_image(outputImg, buffer2, T,R);


    //Escritura del resultado paralelo
    int* buffer3 = (int*)malloc(sizeof(int)*T*R);
    matrix_to_raw(buffer3, H, T, R);
    write_image("parall.raw", buffer3, T,R);
}
