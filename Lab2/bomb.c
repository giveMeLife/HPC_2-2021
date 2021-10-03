#include <stdio.h>
#include <getopt.h>
#include <string.h>
#include <stdlib.h>


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
   
}
