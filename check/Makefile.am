SUBDIRS = .
CLEANFILES = *.temp testc *.o

all: testc

testc: testc.o
	$(MPICC) -o testc testc.o

testc.o: testc.c
	$(MPICC) -c testc.c



