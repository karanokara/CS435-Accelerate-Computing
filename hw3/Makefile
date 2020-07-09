# NOTE: All indented lines are indented with a tab character.
# -G: --device-debug (-G), including "-lineinfo"
# -lineinfo

DFLAGS = -g -G
ZIPNAME = kanra
PROGS = 

all: 1


1: $(PROGS).cu
	nvcc $(PROGS).cu $(DFLAGS) -o $(PROGS) 

clean:
	rm -f $(PROGS)

zip:
	zip $(ZIPNAME).zip Makefile $(PROGS).cu 
