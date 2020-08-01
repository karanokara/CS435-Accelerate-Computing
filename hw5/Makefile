# NOTE: All indented lines are indented with a tab character.
# use for compiling C and CUDA C

ZIPNAME = kanra
CUDAFLAGS = -g -G
CUDAPROG = matrix-multi-cuda
CUDAUNIFIEDPROG = matrix-multi-cuda-unified
CFLAGS = -g -Wall -lm
CPROG = matrix-multi-seq

all: c cuda cuda-unified

c: $(CPROG).c
	gcc $(CPROG).c -o $(CPROG) $(CFLAGS)

cuda: $(CUDAPROG).cu
	nvcc $(CUDAFLAGS) $(CUDAPROG).cu -o $(CUDAPROG)

cuda-unified: $(CUDAUNIFIEDPROG).cu
	nvcc $(CUDAFLAGS) $(CUDAUNIFIEDPROG).cu -o $(CUDAUNIFIEDPROG)

clean:
	rm -f $(CPROG) $(CUDAPROG) $(CUDAUNIFIEDPROG)

zip:
	zip $(ZIPNAME).zip Makefile $(CPROG).c $(CUDAPROG).cu $(CUDAUNIFIEDPROG).cu

run50:
	./${CPROG} 50
	./${CUDAPROG} 50 4 16
	./$(CUDAUNIFIEDPROG) 50 4 16

run100:
	./${CPROG} 100
	./${CUDAPROG} 100 8 16
	./$(CUDAUNIFIEDPROG) 100 8 16

run500:
	./${CPROG} 500
	./${CUDAPROG} 500 50 16
	./$(CUDAUNIFIEDPROG) 500 50 16

run1000:
	./${CPROG} 1000
	./${CUDAPROG} 1000 32 32
	./$(CUDAUNIFIEDPROG) 1000 32 32

run1500:
	./${CPROG} 1500
	./${CUDAPROG} 1500 48 32
	./$(CUDAUNIFIEDPROG) 1500 48 32
