#pragma once

#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <time.h>

__host__ void read_image(char* file_name, unsigned short int * buffer_out, int M, int N);
__global__ void histgmem(unsigned short int* buffer, int* histogram, int image_length);
__global__ void histsmem(unsigned short int* buffer, int* histogram, int image_length);
__global__ void vecadd(float *a, float *b, float *c);
__host__ void debug( int * hist_final, int * hist_final2);
__host__ void write_histogram(char* file_name, int * histogram1, int * histogram2);