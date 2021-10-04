CC= gcc
CFLAGS = -Wall -lm

hough: lectura_escritura.o
	$(CC) -o $@ $^ $(CFLAGS)

lectura_escritura.o: lectura_escritura.c
	$(CC) -c -o $@ $*.c $(CFLAGS)