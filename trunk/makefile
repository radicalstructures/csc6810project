CFLAGS = -Wall -lm
OPT = -O3
CC = gcc 
FILES = main.c functions.c firefly.c 

all :
	$(CC) $(CFLAGS) $(OPT) $(FILES) -o fflies
omp:
	$(CC) $(CFLAGS) $(OPT) -fopenmp $(FILES) -o fflies
ompdebug:
	$(CC) $(CFLAGS) -g -fopenmp $(FILES) -o fflies
debug :
	$(CC) $(CFLAGS) -g $(FILES) -o fflies
test :
	$(CC) -lm -lcunit ./tests/unittests.c -o fflytests
clean :
	rm fflies ./tests/fflytests
