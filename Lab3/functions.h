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
__global__ void histgmem(unsigned short int* buffer, int* histogram);
__global__ void vecadd(float *a, float *b, float *c);