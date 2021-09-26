CC= gcc
CFLAGS = -lm

hough: hough.o
	$(CC) -o $@ $^ $(CFLAGS)

lectura_escritura.o: hough.c
	$(CC) -c -o $@ $*.c $(CFLAGS)


clean:
	rm -rf *.o

run_mul4:
	./hough -i simplehough1-256x256.raw -o salida.raw -M 256 -N 256 -T 512 -R 512 -U 1

run_notmul4:
	./hough -i simplehough1-256x256.raw -o salida.raw -M 256 -N 256 -T 513 -R 512 -U 2