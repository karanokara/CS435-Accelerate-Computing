# NOTE: All indented lines are indented with a tab character.
# -G: --device-debug (-G), including "-lineinfo"
# -lineinfo

DFLAGS = -g -G
PROGS = life_cuda

all: hw2

hw2: $(PROGS).cu
	nvcc $(PROGS).cu $(DFLAGS) -o $(PROGS) 

clean:
	rm -f $(PROGS)

zip:
	zip kanra.zip Makefile $(PROGS).cu
