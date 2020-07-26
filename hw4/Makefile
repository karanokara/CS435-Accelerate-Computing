# NOTE: All indented lines are indented with a tab character.
# use for compiling OpenACC

DFLAGS = -acc -Minfo=accel
ZIPNAME = kanra
PROGS1 = DotProductOpenACC
PROGS2 = IntegrationOpenACC

all: 1 2

1: $(PROGS1).c
	pgcc $(DFLAGS) $(PROGS1).c  -o $(PROGS1) 

2: $(PROGS2).c
	pgcc $(DFLAGS) $(PROGS2).c  -o $(PROGS2) 

clean:
	rm -f $(PROGS1) $(PROGS2)

zip:
	zip $(ZIPNAME).zip Makefile $(PROGS1).c $(PROGS2).c
