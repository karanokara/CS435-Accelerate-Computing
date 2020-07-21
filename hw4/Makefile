# NOTE: All indented lines are indented with a tab character.
# use for compiling OpenACC

DFLAGS = -acc -Minfo=accel
ZIPNAME = kanra
PROGS = openacc

all: 1

1: $(PROGS).c
	pgcc $(DFLAGS) $(PROGS).c  -o $(PROGS) 

clean:
	rm -f $(PROGS)

zip:
	zip $(ZIPNAME).zip Makefile $(PROGS).c
