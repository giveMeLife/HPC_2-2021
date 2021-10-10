#include "functions.h"

int main(int argc, char *argv[]){

  int c;
  int t = 0;
  int N = 0;
  int D = 0;
  char * i = malloc(sizeof(char)*30);
  char * o = malloc(sizeof(char)*30);

  int a;
  while ((c = getopt(argc, argv, "t:N:i:o:D:")) != -1)
    switch (c)
      {
      case 't':
        t = atoi(optarg);
      case 'N':
        N = atoi(optarg);
      case 'i':
        strcpy(i, optarg);
        break;
      case 'o':
        strcpy(o, optarg);
        break;
      case 'D':
        D = atoi(optarg);
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
    

    clock_t start_t, end_t;
    double total_time_sec, total_time_par, total_time_par2, total_time_par3;
    Particle* particles = readFile(i);
    

    /*Prueba de tiempos parte secuencial y paralela*/
    start_t = clock();
    bomb(particles,N);
    end_t = clock();
    total_time_sec = (double)(end_t - start_t) / CLOCKS_PER_SEC;

    float* structure = (float*)malloc(sizeof(float)*N);
    start_t = clock(); 
    structure = bomb_parallel(particles, N, t);
    end_t = clock();
    total_time_par = (double)(end_t - start_t) / CLOCKS_PER_SEC;


    float* structure2 = (float*)malloc(sizeof(float)*N);
    start_t = clock(); 
    structure2 = bomb_parallel2(particles, N, t);
    end_t = clock();
    total_time_par2 = (double)(end_t - start_t) / CLOCKS_PER_SEC;


    float* structure3 = (float*)malloc(sizeof(float)*N);
    start_t = clock(); 
    structure3 = bomb_parallel3(particles, N, t);
    end_t = clock();
    total_time_par3 = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("t_par: %lf, t_par2: %lf, t_par3: %lf, t_sec: %lf\n", total_time_par, total_time_par2, total_time_par3, total_time_sec);

    printf("t_par: %lf, t_par2: %lf, t_sec: %lf\n", total_time_par, total_time_par2, total_time_sec);
    float* structureFinal = (float*)malloc(sizeof(float)*N);
    for (int i = 0; i < N; ++i)
    {
      structureFinal[i] = structure2[i];
    }
    if (D==1){
      niceprint(N,structureFinal);
    }
      

    write_file(o,structure,N);
    
    return 0;
   
}
