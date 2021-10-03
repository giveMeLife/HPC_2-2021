CC= gcc
CFLAGS = -lm -fopenmp

bomb: bomb.o
	$(CC) -o $@ $^ $(CFLAGS)

lectura_escritura.o: bomb.c
	$(CC) -c -o $@ $*.c $(CFLAGS)


clean:
	rm -rf *.o

run:
	./bomb -t 4 -N 10 -i entrada.txt -o output.txt -D 0 