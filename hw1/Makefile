# NOTE: All indented lines are indented with a tab character.

DFLAGS = -g
PROGS= VectorVectorMultiplication

all: hw1

hw1: VectorVectorMultiplication.cu
	nvcc VectorVectorMultiplication.cu $(DFLAGS) -o $(PROGS) 

clean:
	rm -f $(PROGS) a.out

zip:
	zip kanra.zip Makefile VectorVectorMultiplication.cu
