SUBDIRS = .
CLEANFILES = *.o *.a *.so *.la

SRCDIR = $(top_srcdir)/src

all: mympimodule.o mpi.so

mympimodule.o: mympimodule.c
	$(MPICC) $(CFLAGS) -c mympimodule.c -o mympimodule.o

mpi.so: 
	$(MPICC) -shared mympimodule.o $(LDFLAGS) -o mpi.so

#lib_LIBRARIES = libmpi.a

#libmpi_a_SOURCES = mympimodule.c
