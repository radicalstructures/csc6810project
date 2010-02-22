CFLAGS = -Wall
CC = gcc 

all :
	$(CC) main.c firefly.c point.c -lm -fopenmp -o fflies
debug :
	$(CC) main.c firefly.c point.c -lm -g -fopenmp -o fflies
clean :
	rm main.o firefly.o point.o
