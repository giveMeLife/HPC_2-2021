CC= gcc
CFLAGS = -lm -fopenmp 
OBJS_MAIN = bomb.o functions.o
HEADER = functions.h
library = -lniceprint

bomb: $(OBJS_MAIN)
	$(CC) -o bomb $(OBJS_MAIN) $(CFLAGS) $(library) -L .

bomb.o: bomb.c $(HEADER)
	$(CC) -c bomb.c 

functions.o: functions.c $(HEADER)
	$(CC) -c functions.c 

clean:
	rm -rf *.o

run:
	./bomb -t 4 -N 35 -i entrada.txt -o output.txt -D 0 