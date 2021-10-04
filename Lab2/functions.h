#pragma once

#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>

int particles_amount;

struct particle{
    int position;
    double energy;
}; 
typedef struct particle Particle;

void split(char* str, char** words);
Particle* readFile(char* file_name);

