# NOTE: All indented lines are indented with a tab character.
# -G: --device-debug (-G), including "-lineinfo"
# -lineinfo

ZIPNAME = kanra
FLAGSC = -Wall -pg -lm
FLAGSCUDA = -g -G
FLAGSOPENACC = -acc -Minfo=accel
PROGCUDA = life_cuda
PROGCPU = life_seq
PROGOPENACCCPU = life_openacc_cpu
PROGOPENACCGPU = life_openacc_gpu

all: cpu openacc cuda 

cpu: $(PROGCPU).c
	gcc $(PROGCPU).c  -o $(PROGCPU) $(FLAGSC)

openacc: $(PROGOPENACCCPU).c $(PROGOPENACCGPU).c
	pgcc $(FLAGSOPENACC) $(PROGOPENACCCPU).c -o $(PROGOPENACCCPU)
	pgcc $(FLAGSOPENACC) $(PROGOPENACCGPU).c -o $(PROGOPENACCGPU)

cuda: $(PROGCUDA).cu
	nvcc $(FLAGSCUDA) $(PROGCUDA).cu -o $(PROGCUDA)

clean:
	rm -f $(ZIPNAME).zip $(PROGCPU) $(PROGOPENACCCPU) $(PROGOPENACCGPU) $(PROGCUDA) gmon.out

zip:
	zip $(ZIPNAME).zip Makefile $(PROGCPU).c $(PROGOPENACCCPU).c $(PROGOPENACCGPU).c $(PROGCUDA).cu

profile-seq:
	gprof $(PROGCPU) gmon.out

run32:
	./$(PROGCPU) 30 0 32 32
	./$(PROGOPENACCCPU) 30 0 32 32
	./$(PROGOPENACCGPU) 30 0 32 32
	./$(PROGCUDA) 30 0 32 32

run64:
	./$(PROGCPU) 30 0 64 64
	./$(PROGOPENACCCPU) 30 0 64 64
	./$(PROGOPENACCGPU) 30 0 64 64
	./$(PROGCUDA) 30 0 64 64

run128:
	./$(PROGCPU) 30 0 128 128
	./$(PROGOPENACCCPU) 30 0 128 128
	./$(PROGOPENACCGPU) 30 0 128 128
	./$(PROGCUDA) 30 0 128 128

run256:
	./$(PROGCPU) 30 0 256 256
	./$(PROGOPENACCCPU) 30 0 256 256
	./$(PROGOPENACCGPU) 30 0 256 256
	./$(PROGCUDA) 30 0 256 256

run512:
	./$(PROGCPU) 30 0 512 512
	./$(PROGOPENACCCPU) 30 0 512 512
	./$(PROGOPENACCGPU) 30 0 512 512
	./$(PROGCUDA) 30 0 512 512
