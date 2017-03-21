SUBDIRS = .
CLEANFILES = *.o *.so *.a *.la c_ex00 c_ex01 c_ex02 c_ex03 c_ex04 c_ex05 c_ex06 c_ex07 c_ex08 c_ex09 \
c_ex10 c_ex11 c_ex12 c_ex13 f_ex00 f_ex01 f_ex03 f_ex04 f_ex05 f_ex06 f_ex07 f_ex08 f_ex09 \
f_ex10 f_ex11 f_ex12 f_ex13

SRCDIR = $(top_srcdir)/src
LIBDIR = $(top_srcdir)/lib

bin_SCRIPTS = mpi.so

mpi.so: ../src/mpi.so
	cp ../src/mpi.so .

if COMPILE_FORTRAN
  all: c_ex00 c_ex01 c_ex02 c_ex03 c_ex04 c_ex05 c_ex06 c_ex07 c_ex08 c_ex09 c_ex10 c_ex11 c_ex12 c_ex13 \
f_ex00 f_ex01 f_ex03 f_ex04 f_ex05 f_ex06 f_ex07 f_ex08 f_ex09 f_ex10 f_ex11 f_ex12 f_ex13
else
  all: c_ex00 c_ex01 c_ex02 c_ex03 c_ex04 c_ex05 c_ex06 c_ex07 c_ex08 c_ex09 c_ex10 c_ex11 c_ex12 c_ex13 
endif  

c_ex00: c_ex00.c
	$(MPICC) $(CFLAGS) c_ex00.c -o c_ex00

c_ex01: c_ex01.c
	$(MPICC) $(CFLAGS) c_ex01.c -o c_ex01

c_ex02: c_ex02.c
	$(MPICC) $(CFLAGS) c_ex02.c -o c_ex02

c_ex03: c_ex03.c
	$(MPICC) $(CFLAGS) c_ex03.c -o c_ex03

c_ex04: c_ex04.c
	$(MPICC) $(CFLAGS) c_ex04.c -o c_ex04

c_ex05: c_ex05.c
	$(MPICC) $(CFLAGS) c_ex05.c -o c_ex05

c_ex06: c_ex06.c
	$(MPICC) $(CFLAGS) c_ex06.c -o c_ex06

c_ex07: c_ex07.c
	$(MPICC) $(CFLAGS) c_ex07.c -o c_ex07

c_ex08: c_ex08.c
	$(MPICC) $(CFLAGS) c_ex08.c -o c_ex08

c_ex09: c_ex09.c
	$(MPICC) $(CFLAGS) c_ex09.c -o c_ex09

c_ex10: c_ex10.c
	$(MPICC) $(CFLAGS) c_ex10.c -o c_ex10

c_ex11: c_ex11.c
	$(MPICC) $(CFLAGS) c_ex11.c -o c_ex11

c_ex12: c_ex12.c
	$(MPICC) $(CFLAGS) c_ex12.c -o c_ex12

c_ex13: c_ex13.c
	$(MPICC) $(CFLAGS) c_ex13.c -o c_ex13

# MPI FORTRAN Programs

f_ex00: f_ex00.f
	$(MPIFC) $(FFLAGS) f_ex00.f -o f_ex00

f_ex01: f_ex01.f
	$(MPIFC) $(FFLAGS) f_ex01.f -o f_ex01

f_ex03: f_ex03.f
	$(MPIFC) $(FFLAGS) f_ex03.f -o f_ex03

f_ex04: f_ex04.f
	$(MPIFC) $(FFLAGS) f_ex04.f -o f_ex04

f_ex05: f_ex05.f
	$(MPIFC) $(FFLAGS) f_ex05.f -o f_ex05

f_ex06: f_ex06.f
	$(MPIFC) $(FFLAGS) f_ex06.f -o f_ex06

f_ex07: f_ex07.f
	$(MPIFC) $(FFLAGS) f_ex07.f -o f_ex07

f_ex08: f_ex08.f
	$(MPIFC) $(FFLAGS) f_ex08.f -o f_ex08

f_ex09: f_ex09.f
	$(MPIFC) $(FFLAGS) f_ex09.f -o f_ex09

f_ex10: f_ex10.f
	$(MPIFC) $(FFLAGS) f_ex10.f -o f_ex10

f_ex11: f_ex11.f
	$(MPIFC) $(FFLAGS) f_ex11.f -o f_ex11

f_ex12: f_ex12.f
	$(MPIFC) $(FFLAGS) f_ex12.f -o f_ex12

f_ex13: f_ex13.f
	$(MPIFC) $(FFLAGS) f_ex13.f -o f_ex13







