CC = gcc
CFLAGS = -shared -fPIC -O2 -Wall

all: libbfgs.so

libbfgs.so: bfgs_w_varargs.c energy.c
	$(CC) $(CFLAGS) -o libbfgs.so bfgs_w_varargs.c energy.c -lm
