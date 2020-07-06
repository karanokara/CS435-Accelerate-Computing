# NOTE: All indented lines are indented with a tab character.
# -G: --device-debug (-G), including "-lineinfo"
# -lineinfo

DFLAGS = -g -G
PROGS = life_cuda
PROGS2 = life_cuda_tiling

all: 1 2 

2: $(PROGS2).cu
	nvcc $(PROGS2).cu $(DFLAGS) -o $(PROGS2) 

1: $(PROGS).cu
	nvcc $(PROGS).cu $(DFLAGS) -o $(PROGS) 

clean:
	rm -f $(PROGS) $(PROGS2)

zip:
	zip kanra.zip Makefile $(PROGS).cu $(PROGS2).cu
