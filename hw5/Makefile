# NOTE: All indented lines are indented with a tab character.
# use for compiling C and CUDA C

CUDAFLAGS = -g -G
CUDAPROG = matrix-multi-cuda
CFLAGS = -g -Wall -lm
CPROG = matrix-multi-seq

all: c cuda

c: $(CPROG).c
	gcc $(CPROG).c -o $(CPROG) $(CFLAGS)

cuda: $(CUDAPROG).cu
	nvcc $(CUDAFLAGS) $(CUDAPROG).cu -o $(CUDAPROG)

run-500:
	./${CPROG} 500 && ./${CUDAPROG} 500 50 16

clean:
	rm -f $(CPROG) $(CUDAPROG) 