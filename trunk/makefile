CFLAGS = -Wall -lm
OPT = -O3
CC = gcc 

all :
	$(CC) $(CFLAGS) $(OPT) main.c firefly.c    -o fflies
omp:
	$(CC) $(CFLAGS) $(OPT) -fopenmp main.c firefly.c   -o fflies
ompdebug:
	$(CC) $(CFLAGS) -g -fopenmp main.c firefly.c   -o fflies
debug :
	$(CC) $(CFLAGS) -g main.c firefly.c    -o fflies
test :
	$(CC) -lm -lcunit ./tests/unittests.c -o fflytests
clean :
	rm fflies
cleantest :
	rm ./tests/fflytests
