#include "functions.h"


Particle* readFile(char* file_name){

    Particle* particles;
    FILE *fp;
    char *str = (char*)malloc(sizeof(char)*20);
    fp = fopen(file_name, "r");
    if (fp == NULL){
        printf("No se puede abrir archivo: %s\n",file_name);
        exit(1);
    }
    int i = 0;
    while (fgets(str, 20, fp) != NULL){

        if(i == 0){
            particles_amount = atoi(str);
            particles = (Particle*)malloc(sizeof(Particle)*particles_amount);
            i++;
        }
        else{
            int a, end;
            double b;
            if (sscanf(str, "%d%lf %n", &a, &b, &end) == 2 && str[end] == 0) {
            } else {
                fprintf(stderr, "malformed input line: %s", str);
                exit(1);
            }
            particles[i-1].position = a;
            particles[i-1].energy = b;
            i++;
        }
    }
    fclose(fp);
    return particles;
}


void bomb(Particle * particles, int N){
    int pos;
    double energy;
    double MIN_ENERGY = pow(10,-3)/N;
    double * array = (double *)malloc(sizeof(double)*N);
    double value;

    for (int i = 0; i < N; ++i){
        array[i] = 0;
    }

    for (int j = 0; j < particles_amount; ++j){
        pos = particles[j].position;
        energy = particles[j].energy;
        for (int i = 0; i < N; ++i){
            value = array[i] + (1000.0*energy)/(N*sqrt(fabs(pos-i)+1));
            if(value > MIN_ENERGY){
               array[i] = value;
           }
        }

    }

    /*for (int i = 0; i < N; ++i){
        printf("%i: %lf\n",i,array[i]);
    }*/
}

double* bomb_parallel(Particle * particles, int N, int t){
        double *final_array = (double *)malloc(sizeof(double)*N);
        double MIN_ENERGY = pow(10,-3)/N; 

        for (int i = 0; i < N; i++){
            final_array[i] = 0;
        }
        int contador = 0;
    #pragma omp parallel shared(final_array, N, particles) num_threads(t)
    {
        double value;
        int i;
        int j;
        int pos;
        double energy;
        double * array;
        array = (double *)malloc(sizeof(double)*N);
        for ( i = 0; i < N; ++i){
            array[i] = 0;
        }


        #pragma omp for schedule(dynamic, 2)
            for (j = 0; j < particles_amount; j++){
                pos = particles[j].position;
                energy = particles[j].energy;
                for (i = 0; i < N; i++){
                    value =array[i] + (1000.0*energy)/(N*sqrt(fabs(pos-(double)i)+1));
                        if(value  > MIN_ENERGY){
                            array[i] =   value;
                        }
                    
                }
            }
       #pragma omp critical
        {        
            for( j = 0; j<N; j++){   
                final_array[j] += array[j];
            }
        }
        
    }
    
    for (int i = 0; i < N; i++){
        printf("%i: %lf\n",i,final_array[i]);
    }
    return(final_array);
}

double* bomb_parallel2(Particle * particles, int N, int t){
        double *final_array = (double *)malloc(sizeof(double)*N);
        double MIN_ENERGY = pow(10,-3)/N; 

        for (int i = 0; i < N; i++){
            final_array[i] = 0;
        }
        int contador = 0;
    #pragma omp parallel shared(final_array, N, particles) num_threads(t)
    {
        double value;
        int i;
        int j;
        int pos;
        double energy;


        #pragma omp for schedule(dynamic, 2)
            for (j = 0; j < particles_amount; j++){
                pos = particles[j].position;
                energy = particles[j].energy;
                for (i = 0; i < N; i++){
                    value = (1000.0*energy)/(N*sqrt(fabs(pos-(double)i)+1));
                    #pragma omp critical
                    {
                        if(value + final_array[i]  > MIN_ENERGY){
                            final_array[i] += value;
                        }
                    }
                    
                }
            }
        
    }
    
    return(final_array);
}

double * bomb_parallel3(Particle * particles, int N, int t){
    int pos;
    double energy;
    double MIN_ENERGY = pow(10,-3)/N;
    double * array = (double *)malloc(sizeof(double)*N);
    double value;

    for (int i = 0; i < N; ++i){
        array[i] = 0;
    }

    for (int j = 0; j < particles_amount; ++j){
        pos = particles[j].position;
        energy = particles[j].energy;
        #pragma omp parallel shared(array) private(value) num_threads(t)
        {
            #pragma omp for schedule(dynamic, 2)
            for (int i = 0; i < N; ++i){
               // printf("Soy la hebra %d y ejecuto i: %d \n",omp_get_thread_num(),i);
                value = array[i] + (1000.0*energy)/(N*sqrt(fabs(pos-i)+1));
                if(value > MIN_ENERGY){
                   array[i] = value;
               }
            }
        }
    }
    return(array);
}

double* maximum_energy(double* energies, int N){
    double* values = (double*)malloc(sizeof(double)*2);
    int index = -1;
    double max_energy = 0.0, actual_energy;
    for(int i = 0; i<N; i++){
        actual_energy = energies[i];
        if(actual_energy>max_energy){
            max_energy = actual_energy;
            index = i;
        }
    }
    values[0] = (double)index;
    values[1] = max_energy;
    return(values);
}

void write_file(char* file_name, double*energies, int N){
    FILE *out_file = fopen(file_name, "w");
    int i = 0;
    double* max = (double*)malloc(sizeof(double)*2);
    max = maximum_energy(energies, N);
    while(i<N+1){
        if(i == 0){
            fprintf(out_file, "%d %lf\n", (int)max[0], max[1]);
            i=i+1;
        }
        else{
            fprintf(out_file, "%d %lf\n", i-1, energies[i-1]);
            i=i+1;
        }
    }
    fclose(out_file);
}