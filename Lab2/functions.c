#include "functions.h"


void split(char* str, char** words){
    int j = 0;
    int k = 0;
    for(int i = 0; i<strlen(str); i++){
        if(str[i] == ' '){
            j++;
            k = 0;
        }
        else if(str[i] == '\n'){
            i = strlen(str);
        }
        else{
            words[j][k] = str[i];
            k++;
        }
    }
}

Particle* readFile(char* file_name){
    Particle* particles;
    FILE *fp;
    char *str = (char*)malloc(sizeof(char)*20);
    char** line = (char**)malloc(sizeof(char*)*2);
    line[0] = (char*)malloc(sizeof(char)*20);
    line[1] = (char*)malloc(sizeof(char)*20);
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
            split(str,line);
            particles[i-1].position = atoi(line[0]);
            particles[i-1].energy = strtod(line[1], NULL);
            free(line[0]); free(line[1]);
            line[0] = (char*)malloc(sizeof(char)*20); line[1] = (char*)malloc(sizeof(char)*20);
            i++;
        }
    }
    free(line[0]); free(line[1]);    
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

    for (int i = 0; i < N; ++i){
        printf("%i: %lf\n",i,array[i]);
    }
}