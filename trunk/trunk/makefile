CFLAGS = -Wall -fopenmp -lm
CC = gcc 

all :
	$(CC) $(CFLAGS) main.c firefly.c point.c  -o fflies
debug :
	$(CC) $(CFLAGS) main.c firefly.c point.c  -g -o fflies
clean :
	rm main.o firefly.o point.o
