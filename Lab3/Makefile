CC= nvcc
CFLAGS = -lm  
OBJS_MAIN = hist.o functions.o
HEADER = functions.h
  
hist: $(OBJS_MAIN)
	$(CC) -o hist $(OBJS_MAIN) $(CFLAGS)

hist.o: hist.cu $(HEADER)
	$(CC) -c hist.cu 

functions.o: functions.cu $(HEADER)
	$(CC) -c functions.cu 

clean:
	rm -rf *.o

run:
	./hist -i img2lab3-512x512.raw -m 512 -n 512 -o histogram.txt -t 1 -t 512 -d 