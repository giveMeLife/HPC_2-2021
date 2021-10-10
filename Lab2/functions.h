#pragma once

#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <time.h>

extern void niceprint(int N, float *Energy);

int particles_amount;
struct particle{
    int position;
    float energy;
}; 
typedef struct particle Particle;


Particle* readFile(char* file_name);
void bomb(Particle * particles, int N);
float* bomb_parallel(Particle * particles, int N, int t);
float* bomb_parallel2(Particle * particles, int N, int t);
float* bomb_parallel3(Particle * particles, int N, int t);

float* maximum_energy(float* energies, int N);
void write_file(char* file_name, float*energies, int N);
float* maximum_energy(float* energies, int N);
