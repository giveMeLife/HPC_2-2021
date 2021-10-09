CC= gcc
CFLAGS = -lm -fopenmp -lniceprint -L
headers = function.h
library = lniceprint
bomb: bomb.o function.o
	$(CC) -o bomb.o function.o $(CFLAGS) 

bomb.o: bomb.c 
	$(CC) -c bomb.c 

function.o: function.c 
	$(CC) -c function 

clean:
	rm -rf *.o

run:
	./bomb -t 4 -N 35 -i entrada.txt -o output.txt -D 0 