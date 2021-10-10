#include "functions.h"



/*
Descripción: Función que se encarga de leer archivo de entrada con la cantidad de partículas a impactar, y la posición de impacto
             de cada partícula, junto con su energía.
Entrada: Nombre del archivo de entrada ("file_name")
Proceso: Se obtiene línea por línea del archivo y la primera línea se guarda como la cantidad de partículas, 
         y el resto se guardan en un arreglo de estructuras que tienen la información de cada partícula.  
Salida: Como salida se genera un arreglo de estructuras Particle, que contiene la información de cada partícula (Posición de impacto
        y energía)
*/
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
            float b;
            if (sscanf(str, "%d%f %n", &a, &b, &end) == 2 && str[end] == 0) {
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
    float energy;
    float MIN_ENERGY = pow(10,-3)/N;
    float * array = (float *)malloc(sizeof(float)*N);
    float value;

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
}

/*
Descripción: Función que se encarga de recorrer el arreglo de particulas de manera paralela, calcular el efecto
             del choque de estas en el material
Entrada: Estructura que contiene a cada una de las partículas del archivo de entrada con la posición de impacto y 
         la energía. Entero que indica la cantidad de particulas que tiene el material y número de hebras. 
Proceso: Se crea un arreglo que almacena el estado final del material. Luego se inicializa el bloque con omp, en donde el arreglo final es
         una variable compartida, al igual que el número de elementos que este contiene y la estructura de particulas. A continuación, cada una
         de las hebras calcula el impacto que tienen las particulas que ingresan en el material y lo almacena en un arreglo privado, la división
         de las hebras se hace mediante schedule, en donde se indica que estas de manera dinámica realizarán dos de las iteraciones del for, es
         decir, cada una de las hebras realiza en la medida de lo posible el cálculo del impacto de dos partículas en el material.  Finalmente 
         se declara una sección crítica que se encarga de sumar en el arreglo principal los arreglos privados de cada hebra.
Salida: Como salida se obtiene un arreglo de flotantes que contiene la energía final en el material.
*/
float* bomb_parallel(Particle * particles, int N, int t){
        float *final_array = (float *)malloc(sizeof(float)*N);
        float MIN_ENERGY = pow(10,-3)/N; 

        for (int i = 0; i < N; i++){
            final_array[i] = 0;
        }
        int contador = 0;
    #pragma omp parallel shared(final_array, N, particles) num_threads(t)
    {
        float value;
        int i;
        int j;
        int pos;
        float energy;
        float * array;
        array = (float *)malloc(sizeof(float)*N);
        for ( i = 0; i < N; ++i){
            array[i] = 0;
        }


        #pragma omp for schedule(dynamic, 2)
            for (j = 0; j < particles_amount; j++){
                pos = particles[j].position;
                energy = particles[j].energy;
                for (i = 0; i < N; i++){
                    value =array[i] + (1000.0*energy)/(N*sqrt(fabs(pos-(float)i)+1));
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

    return(final_array);
}

/*
Descripción: Función que se encarga de recorrer el arreglo de particulas de manera paralela, calcular el efecto
             del choque de estas en el material
Entrada: Estructura que contiene a cada una de las partículas del archivo de entrada con la posición de impacto y 
         la energía. Entero que indica la cantidad de particulas que tiene el material y número de hebras. 
Proceso: Se crea un arreglo que almacena el estado final del material. Luego se inicializa el bloque con omp, en donde el arreglo final es
         una variable compartida, al igual que el número de elementos que este contiene y la estructura de particulas. A continuación, cada una
         de las hebras calcula el impacto que tienen las particulas que ingresan en el material y lo almacena en un arreglo compartido, la división
         de las hebras se hace mediante schedule, en done se indican que estas de manera dinámica realizarán dos de las iteraciones del for, es
         decir, cada una de las hebras realiza en la medida de lo posible el cálculo del impacto de dos partículas en el material.  A diferencia de 
         la implementación anterior la sección crítica ocurre en el arreglo que almacena el estado final del material.
Salida: Como salida se obtiene un arreglo de flotantes que contiene la energía final en el material.
*/
float* bomb_parallel2(Particle * particles, int N, int t){
        float *final_array = (float *)malloc(sizeof(float)*N);
        float MIN_ENERGY = pow(10,-3)/N; 

        for (int i = 0; i < N; i++){
            final_array[i] = 0;
        }
        int contador = 0;
    #pragma omp parallel shared(final_array, N, particles) num_threads(t)
    {
        float value;
        int i;
        int j;
        int pos;
        float energy;


        #pragma omp for schedule(dynamic, 2)
            for (j = 0; j < particles_amount; j++){
                pos = particles[j].position;
                energy = particles[j].energy;
                for (i = 0; i < N; i++){
                    value = (1000.0*energy)/(N*sqrt(fabs(pos-(float)i)+1));
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

/*
Descripción: Función que se encarga de recorrer el arreglo de particulas de manera paralela, calcular el efecto
             del choque de estas en el material
Entrada: Estructura que contiene a cada una de las partículas del archivo de entrada con la posición de impacto y 
         la energía. Entero que indica la cantidad de particulas que tiene el material y número de hebras. 
Proceso: Se crea un arreglo que almacena el estado final del material y que es de tipo compartido y se define el cálculo
         de la energía que genera el impacto de una partícula para cada posición del material como privado (value). Luego 
         se comienza el for paralelo, el cual genera que cada una de las hebras calcule 2 posiciones del for para cada una 
         de las particulas. En este caso no existe condición de carrera, ya que cada hebra solo accede a una posición i, por
         lo que no se pueden sobreescribir los valores.
Salida: Como salida se obtiene un arreglo de flotantes que contiene la energía final en el material.
*/
float * bomb_parallel3(Particle * particles, int N, int t){
    int pos;
    float energy;
    float MIN_ENERGY = pow(10,-3)/N;
    float * array = (float *)malloc(sizeof(float)*N);
    float value;

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
                value = array[i] + (1000.0*energy)/(N*sqrt(fabs(pos-i)+1));
                if(value > MIN_ENERGY){
                   array[i] = value;
               }
            }
        }
    }
    return(array);
}


/*
Descripción: Función que se encarga de determinar la energía máxima almacenada en la estructura luego del bombardeo de
             partículas.
Entrada: Arreglo de tipo double, donde se tienen las energías de la estructura y un entero N indicando el largo del arreglo.
Proceso: Se recorre el arreglo preguntando si el dato analizado es el máximo dentro del arreglo.
Salida: Un arreglo de double de tamaño 2, indicando el primero la posición dentro de la estructura y el segundo, la máxima energía.
*/
float* maximum_energy(float* energies, int N){
    float* values = (float*)malloc(sizeof(float)*2);
    int index = -1;
    float max_energy = 0.0, actual_energy;
    for(int i = 0; i<N; i++){
        actual_energy = energies[i];
        if(actual_energy>max_energy){
            max_energy = actual_energy;
            index = i;
        }
    }
    values[0] = (float)index;
    values[1] = max_energy;
    return(values);
}

/*
Descripción: Función que se encarga de generar un archivo de salida indicando la energía almacenada en cada posición del material, y
             la posición que contiene la máxima energía junto con esta.
Entrada: String que indica el nombre del archivo de salida, Arreglo de tipo double donde se tienen las energías de la estructura 
         y un entero N indicando el largo del arreglo.
Proceso: Se agrega consecutivamente las líneas a un archivo de salida con dos columnas por línea, indicando posición en el material y energía
         almacenada.
Salida: -
*/
void write_file(char* file_name, float*energies, int N){
    FILE *out_file = fopen(file_name, "w");
    int i = 0;
    float* max = (float*)malloc(sizeof(float)*2);
    max = maximum_energy(energies, N);
    while(i<N+1){
        if(i == 0){
            fprintf(out_file, "%d %f\n", (int)max[0], max[1]);
            i=i+1;
        }
        else{
            fprintf(out_file, "%d %f\n", i-1, energies[i-1]);
            i=i+1;
        }
    }
    fclose(out_file);
}